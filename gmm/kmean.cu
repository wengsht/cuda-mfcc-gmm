#include "kmean.h"

#include "Feat_comm.h"
#include "memory_srlr.h"
#include "kmean_kernel.h"

#include "wtime.h"

#define LOG2PI (log(2*PI))

const int N = 1000000;
KMean::KMean() {
    m_WeightIndex = NULL;
    
    dimObjects = NULL;
    memberShip = NULL;
    device_memberShip = NULL;
    device_dimObjects = NULL;
    
    clusterSize = NULL;
    device_clusterSize = NULL;
    
    dimClusters = NULL;
    device_dimClusters = NULL;
    
    dimDiagCovs = NULL;
    device_diagCovs = NULL;
    
    dMats = NULL;
    device_dMats = NULL;
}

KMean::~KMean() {
    /*
    if(m_WeightIndex) {
        free(m_WeightIndex);
        m_WeightIndex=NULL;	
    }
    */
    
    Free2D(dimObjects);
    Free2D(dimClusters);
    
    Free(memberShip);
    
    if(device_dimObjects) {
        cudaFree(device_dimObjects);
        device_dimObjects = NULL;
    }
    
    if(device_memberShip) {
        cudaFree(device_memberShip);
        device_memberShip = NULL;
    }
    
    if(device_clusterSize) {
        cudaFree(device_clusterSize);
        device_clusterSize = NULL;
    }
    Free(clusterSize);
    
    Free2D(dimDiagCovs);

    if(device_dimClusters) {
        cudaFree(device_dimClusters);
        device_dimClusters = NULL;
    }
    
    if(device_diagCovs) {
        cudaFree(device_diagCovs);
        
        device_diagCovs = NULL;
    }
    
    Free(dMats);
    if(device_dMats) {
        cudaFree(device_dMats);
        device_dMats = NULL;
    }
}

void KMean::KMeanCluster() {
    int numOfBlock = (numObjs - 1) / FIND_NEAREST_BLOCKSIZE + 1;
    int shareMemorySize = numDims * numClusters * sizeof(float);

    if(shareMemorySize > 49152) {
        shareMemorySize = 1;
    }
    
    cuda_find_nearest_cluster<<<numOfBlock, FIND_NEAREST_BLOCKSIZE, shareMemorySize>>>(shareMemorySize, device_dimObjects, device_dimClusters, numObjs, numDims, numClusters, device_memberShip);
    
    numOfBlock = (numObjs - 1) / ACCUMULATE_BLOCKSIZE + 1;
    cudaMemset(device_dimClusters, 0, numClusters * numDims * sizeof(float));
    cudaMemset(device_clusterSize, 0, numClusters * sizeof(int));
    
    cuda_accumulate_clusters<<<numOfBlock, ACCUMULATE_BLOCKSIZE>>>(device_dimObjects, device_memberShip, numObjs, numDims, numClusters, device_clusterSize, device_dimClusters);
    
    cuda_average_clusters<<<numDims, numClusters>>> (device_clusterSize, device_dimClusters);
}


void KMean::DataPrepare(struct Features &features, int MaxMixNum) {
    this->features = &features;

    numDims = features.featureDim;

    /// COPY features into buffers
    numObjs = 0;
    for(int idx = 0;idx < features.nFeatures;idx++) {
        numObjs += features.featureSize[idx];
    }

    Malloc2D(dimObjects, numDims, numObjs, float);

    memberShip = (int *) malloc(sizeof(int) * numObjs);
    cudaMalloc(&device_memberShip, sizeof(int) * numObjs);

    int featureIdx = 0;
    for(int idx = 0;idx < features.nFeatures; idx++) {
        for(int idy = 0; idy < features.featureSize[idx]; idy ++) {
            for(int idz = 0;idz < numDims; idz ++) {
                dimObjects[idz][featureIdx] = features.features[idx][idy * numDims + idz];
            }
            featureIdx++;
        }
    }

    Malloc2D(dimClusters, numDims, MaxMixNum, float);

    cudaMalloc(&device_dimObjects, numObjs * numDims * sizeof(float));
    cudaMalloc(&device_dimClusters, MaxMixNum * numDims * sizeof(float));

    Malloc2D(dimDiagCovs, numDims, MaxMixNum, float);
    cudaMalloc(&device_diagCovs, MaxMixNum * numDims * sizeof(float));

    cudaMemcpy(device_dimObjects, dimObjects[0], numObjs * numDims * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&device_clusterSize, MaxMixNum * sizeof(int));
    clusterSize = (int *) malloc(MaxMixNum * sizeof(int));

    dMats = (float *) malloc(MaxMixNum * sizeof(float));
    cudaMalloc(&device_dMats, MaxMixNum * sizeof(float));
}

void KMean::KMeanInitalize() {
    numClusters = m_nMixNum;

    for(int j=0;j < numDims;j++) {
        for(int m=0;m < numClusters;m++) {
            dimClusters[0][j*numClusters + m] = m_pGmmModel[0].pGauss[m].pfMean[j];
        }
    }
    
    cudaMemcpy(device_dimClusters, dimClusters[0], numClusters * numDims * sizeof(float), cudaMemcpyHostToDevice);
}

void KMean::KMeanFinalize() {
    // clusters
    cudaMemcpy(dimClusters[0], device_dimClusters, numClusters * numDims * sizeof(float), cudaMemcpyDeviceToHost);

    // clusterSize
    cudaMemcpy(clusterSize, device_clusterSize, numClusters * sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int gaussIdx = 0; gaussIdx < numClusters; gaussIdx++) {
        m_pGmmModel[0].pfWeight[gaussIdx] = (float)clusterSize[gaussIdx] / numObjs;
    }

    /// diag Covariance 
    int numOfBlock = (numObjs - 1) / ACCUMULATE_BLOCKSIZE + 1;
    cudaMemset(device_diagCovs, 0, numClusters * numDims * sizeof(float));

    cuda_accumulate_diagcovs<<<numOfBlock, ACCUMULATE_BLOCKSIZE>>>(device_dimObjects, device_memberShip, device_dimClusters, numObjs, numDims, numClusters, device_diagCovs);

    cudaMemset(device_dMats, 0, numClusters * sizeof(float));

    cuda_average_diagcovs<<<numDims, numClusters>>>(device_clusterSize, device_diagCovs, device_dMats);

    cudaMemcpy(dMats, device_dMats, numClusters * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(dimDiagCovs[0], device_diagCovs, numClusters * numDims * sizeof(float), cudaMemcpyDeviceToHost);

    for(int clusterIdx = 0;clusterIdx < numClusters; clusterIdx++) {
        m_pGmmModel[0].pGauss[clusterIdx].dMat = (dMats[clusterIdx] - numDims * LOG2PI) * 0.5 + log(m_pGmmModel[0].pfWeight[clusterIdx]);

        for(int dimIdx = 0; dimIdx < numDims; dimIdx ++) {
            m_pGmmModel[0].pGauss[clusterIdx].pfDiagCov[dimIdx] = dimDiagCovs[0][dimIdx * numClusters + clusterIdx]; 
            m_pGmmModel[0].pGauss[clusterIdx].pfMean[dimIdx] = dimClusters[0][dimIdx * numClusters + clusterIdx];
        }
    }
}

void KMean::KMeanIteration() {
    KMeanCluster();
}

void KMean::KMeanMain(int KMeanIterNum) {
    /// New Splited Mixture Gaussian Model
    KMeanInitalize();
    
    for(int i = 0;i < KMeanIterNum;i++) {
        KMeanIteration();
    }

    KMeanFinalize();
}
