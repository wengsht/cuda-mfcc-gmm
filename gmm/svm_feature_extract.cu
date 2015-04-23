// =====================================================================================
// 
//       Filename:  svm_feature_extract.cpp
// 
//    Description:  
// 
//        Version:  0.01
//        Created:  04/21/2015 09:58:18 PM
//       Revision:  none
//       Compiler:  clang 3.5
// 
//         Author:  wengsht (SYSU-CMU), wengsht.sysu@gmail.com
//        Company:  
// 
// =====================================================================================

#include <stdio.h>
#include <string.h>

#include "GMMParam.h"
#include "Feat_comm.h"
#include "em_kernel.h"

#define HUMAN_TAG "human"
#define SPOOF_TAG "spoof"

void svm_normalize(float *mgpp, int train_file_count, int all_file_count, int numClusters);
void write_svm_features(float *mgpp, int *tags, int nFiles, int nClusters, char *svm_filename);
void read_tags(char *tag_filename, int *tags);

#define VecSize 36
#define NFILE_PER_CUDA 1000
#define MIN_F -1
#define MAX_F 1

int main(int argc, char **argv) {
    if(argc < 5) {
        printf("usage: ./svm_feature_extract TRAINED_GMM_MODEL waglist taglist train_count"); // tagfile
    }

    char svm_train_output_file[] = "svm_train.feature";
    char svm_test_output_file[] = "svm_test.feature";
    
    int train_file_count;
    sscanf(argv[4], "%d", &train_file_count);
    
    GMMParam gmm_model;
    gmm_model.LoadModel(argv[1]);

    struct Features mfccs;
    int maxFrame;
    ReadFeatures(argv[2], mfccs, maxFrame, VecSize);

    int numDims = VecSize;
    int numClusters = gmm_model.GetMixtureNum();

    /// COPY THE mean var to cuda
    GaussMixModel * m_pGmmModel = gmm_model.GetRawMixModel();

    float **dimObjects=NULL, *device_dimObjects = NULL;

    float **dimClusters = NULL;
    float ** dimDiagCovs = NULL;
    float  *dMats = NULL;
    float *device_dimClusters = NULL, * device_diagCovs = NULL, * device_dMats = NULL;
    
    float **probabilities = NULL;
    float *device_probabilities = NULL;

    Malloc2D(dimClusters, numDims, numClusters, float);
    cudaMalloc(&device_dimClusters, numClusters * numDims * sizeof(float));

    Malloc2D(dimDiagCovs, numDims, numClusters, float);
    cudaMalloc(&device_diagCovs, numClusters * numDims * sizeof(float));

    dMats = (float *) malloc(numClusters * sizeof(float));
    cudaMalloc(&device_dMats, numClusters * sizeof(float));

    float *mgpp = (float *) malloc(mfccs.nFeatures * numClusters * sizeof(float));
    int *tags = (int *) malloc(sizeof(int) * mfccs.nFeatures);

    Malloc2D(dimObjects, VecSize, NFILE_PER_CUDA*maxFrame, float);
    cudaMalloc(&device_dimObjects, NFILE_PER_CUDA*maxFrame* numDims * sizeof(float));

    Malloc2D(probabilities, NFILE_PER_CUDA * maxFrame, numClusters, float);
    cudaMalloc(&device_probabilities, NFILE_PER_CUDA * maxFrame * numClusters* sizeof(float));

    // read in tags  
    read_tags(argv[3], tags);
    
    // load models !!!
    for(int m=0;m < numClusters;m++) {
        dMats[m] = m_pGmmModel[0].pGauss[m].dMat;
        for(int j=0;j < numDims;j++) {
            dimClusters[0][j*numClusters + m] = m_pGmmModel[0].pGauss[m].pfMean[j];
            dimDiagCovs[0][j*numClusters + m] = m_pGmmModel[0].pGauss[m].pfDiagCov[j];
        }
    }

    cudaMemcpy(device_dimClusters, dimClusters[0], numClusters * numDims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_diagCovs, dimDiagCovs[0], numClusters * numDims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dMats, dMats, numClusters * sizeof(float), cudaMemcpyHostToDevice);
    // features!!

    memset(mgpp, 0, sizeof(float) * numDims * mfccs.nFeatures);
    for(int i = 0;i < mfccs.nFeatures; i += NFILE_PER_CUDA) {
        int numObjs= 0;

        for(int idx = i; idx < mfccs.nFeatures && idx < i + NFILE_PER_CUDA; idx++) {
            for(int idy = 0; idy < mfccs.featureSize[idx]; idy ++) {
                for(int idz = 0;idz < numDims; idz ++) {
                    dimObjects[idz][numObjs] = mfccs.features[idx][idy * numDims + idz];
                }
                numObjs++;
            }
        }
        cudaMemcpy(device_dimObjects, dimObjects[0], numObjs * numDims * sizeof(float), cudaMemcpyHostToDevice);

        int numBlocks = (numObjs - 1) / OBSERVATE_BLOCK_SIZE + 1;
        int sharedSize = 2 * numDims * numClusters * sizeof(float) +  numClusters * sizeof(float);

        if(sharedSize > 49152) {
            sharedSize = 1;
        }

        cuda_observate_probability<<<numBlocks, OBSERVATE_BLOCK_SIZE, sharedSize>>>(sharedSize, device_dimObjects, device_dimClusters, device_diagCovs, device_dMats, numObjs, numDims, numClusters, device_probabilities);

        cudaMemcpy(probabilities[0], device_probabilities, sizeof(float) * numObjs * numClusters, cudaMemcpyDeviceToHost);

        int featureIdx = 0;
        for(int idx = i; idx < mfccs.nFeatures && idx < i + NFILE_PER_CUDA; idx++) {
            int fileSize = mfccs.featureSize[idx];
            for(int idy = 0; idy < fileSize; idy ++) {
                for(int idz = 0;idz < numClusters; idz ++) {
                    mgpp[idx*numClusters + idz] += probabilities[0][featureIdx*numClusters + idz];
                }
                featureIdx ++;
            }
            for(int idz = 0;idz < numClusters; idz ++) {
                float val = sqrt(mgpp[idx*numClusters + idz] / fileSize);

                mgpp[idx * numClusters + idz] = val;
                
            }
        }
    }
    
    // [-1 1] normalization
    svm_normalize(mgpp, train_file_count, mfccs.nFeatures, numClusters);
    
    write_svm_features(mgpp, tags, train_file_count, numClusters, svm_train_output_file);
    write_svm_features(mgpp + train_file_count * numClusters, tags + train_file_count, mfccs.nFeatures - train_file_count, numClusters, svm_test_output_file);
    
    /// Exiting

    if(device_dimObjects) {
        cudaFree(device_dimObjects);
        device_dimObjects = NULL;
    }
    Free2D(dimObjects);

    Free2D(dimClusters);
    if(device_dimClusters) {
        cudaFree(device_dimClusters);
        device_dimClusters = NULL;
    }
    Free2D(dimDiagCovs);

    if(device_diagCovs) {
        cudaFree(device_diagCovs);

        device_diagCovs = NULL;
    }

    Free(dMats);
    if(device_dMats) {
        cudaFree(device_dMats);
        device_dMats = NULL;
    }
    Free2D(probabilities);
    if(device_probabilities) {
        cudaFree(device_probabilities);
        device_probabilities = NULL;
    }
    if(mgpp) {
        free(mgpp);
        mgpp = NULL;
    }
    FreeFeatures(mfccs);
    
    if(tags) {
        free(tags);
        tags = NULL;
    }

    return 0;
}

void read_tags(char *tag_filename, int *tags) {
    FILE *fp;
    
    fp = fopen(tag_filename, "r");
    
    if(! fp) {
        printf("file open error\n");
    }
    
    int size;
    char buf[100];
    fscanf(fp, "%d\n", &size);
    for(int i = 0;i < size;i++) {
        fscanf(fp, "%s", buf);
        
        if(strcmp(HUMAN_TAG, buf) == 0) {
            tags[i] = 1;
        }
        else {
            tags[i] = -1;
        }
    }
    fclose(fp);
}

void write_svm_features(float *mgpp, int *tags, int nFiles, int nClusters, char *svm_filename) {
    FILE *fp;
    
    fp = fopen(svm_filename, "w");
    for(int i = 0;i < nFiles; i++) {
        if(tags[i] == 1) 
            fprintf(fp, "+1");
        else 
            fprintf(fp, "-1");
        
        for(int j = 1; j <= nClusters; j++) {
            fprintf(fp, " %d:%f", j, mgpp[i*nClusters + j-1]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void svm_normalize(float *mgpp, int train_file_count, int all_file_count, int numClusters) {
    float *max_f = (float *)malloc(numClusters * sizeof(float));
    float *min_f = (float *)malloc(numClusters * sizeof(float));
    
    memcpy(max_f, mgpp, sizeof(float) * numClusters);
    memcpy(min_f, mgpp, sizeof(float) * numClusters);
    
    for(int i = 1; i < train_file_count; i++) {
        for(int j = 0;j < numClusters;j++) {
            if(max_f[j] < mgpp[i*numClusters + j]) {
                max_f[j] = mgpp[i*numClusters + j];
            }
            if(min_f[j] > mgpp[i*numClusters + j]) {
                min_f[j] = mgpp[i*numClusters + j];
            }
        }
    }
    
    for(int i = 0;i < all_file_count; i++) {
        for(int j = 0;j < numClusters;j++) {
            mgpp[i*numClusters+j] = (mgpp[i*numClusters+j] - min_f[j]) * (MAX_F - MIN_F) / (max_f[j] - min_f[j]) + MIN_F;
        }
    }

    free(max_f);
    free(min_f);
}
