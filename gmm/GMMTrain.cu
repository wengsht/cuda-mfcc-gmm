#include "GMMTrain.h"

#include <cstdlib>
#include <math.h>
#include "assert.h"

#include "em_kernel.h"
#include "Feat_comm.h"
#include "memory_srlr.h"
#include "wtime.h"

#define e 2.718281828459

GMMTrain::GMMTrain() {
    m_fWeightFloor = -1.f;		// 表示无效
    m_pfVarFloor = NULL;
    m_fOccFloor = -1.f;

    // cuda 
    dimObjects = NULL;
    device_dimObjects = NULL;
    
    dimClusters = NULL;
    device_dimClusters = NULL;
    
    dimDiagCovs = NULL;
    device_diagCovs = NULL;

    dMats = NULL;
    device_dMats = NULL;
    
    dOcc = NULL;
    device_reduceOcc[0] = NULL;
    device_reduceOcc[1] = NULL;
    
    device_meanOcc[0] = NULL;
    device_meanOcc[1] = NULL;
    meanOcc = NULL;
    
    device_varOcc[0] = NULL;
    device_varOcc[1] = NULL;
    varOcc = NULL;
}


void GMMTrain::DataPrepare(struct Features &features ,int MaxMixNum) {
    this->features = &features;

    /// COPY features into buffers
    numDims = features.featureDim;
    numObjs = 0;
    for(int idx = 0;idx < features.nFeatures;idx++) {
        numObjs += features.featureSize[idx];
    }
    Malloc2D(dimObjects, numDims, numObjs, float);
    //printf("wengshtxx %d %d %d\n", numObjs, MaxMixNum, numDims);
    
//    printf("xxd %d\n", numObjs);
    int featureIdx = 0;
    for(int idx = 0;idx < features.nFeatures; idx++) {
        for(int idy = 0; idy < features.featureSize[idx]; idy ++) {
            for(int idz = 0;idz < numDims; idz ++) {
                dimObjects[idz][featureIdx] = features.features[idx][idy * numDims + idz];
            }
            featureIdx++;
        }
    }
    
    cudaMalloc(&device_dimObjects, numObjs * numDims * sizeof(float));
    cudaMemcpy(device_dimObjects, dimObjects[0], numObjs * numDims * sizeof(float), cudaMemcpyHostToDevice);

    Malloc2D(dimClusters, numDims, MaxMixNum, float);
    cudaMalloc(&device_dimClusters, MaxMixNum * numDims * sizeof(float));

    Malloc2D(dimDiagCovs, numDims, MaxMixNum, float);
    cudaMalloc(&device_diagCovs, MaxMixNum * numDims * sizeof(float));
    dMats = (float *) malloc(MaxMixNum * sizeof(float));
    cudaMalloc(&device_dMats, MaxMixNum * sizeof(float));

    Malloc2D(probabilities, numObjs, MaxMixNum, float);
    cudaMalloc(&device_probabilities, numObjs * MaxMixNum * sizeof(float));
    
    dOcc = (float *) malloc(sizeof(float) * MaxMixNum);
    
    device_reduceOcc[0] = device_probabilities;
//    if(numObjs > OCC_REDUCE_BLOCK_SIZE)
    cudaMalloc(&device_reduceOcc[1], (numObjs/OCC_REDUCE_BLOCK_SIZE+1) * MaxMixNum * sizeof(float));

    cudaMalloc(&device_meanOcc[0], (numObjs /OCC_MEANVAR_REDUCE_BLOCK_SIZE+ 1) * numDims * MaxMixNum * sizeof(float));
    cudaMalloc(&device_meanOcc[1], ( ((numObjs/OCC_MEANVAR_REDUCE_BLOCK_SIZE+1)/OCC_MEANVAR_REDUCE_BLOCK_SIZE+1) * numDims * MaxMixNum * sizeof(float)));

    cudaMalloc(&device_varOcc[0], (numObjs /OCC_MEANVAR_REDUCE_BLOCK_SIZE+ 1) * numDims * MaxMixNum * sizeof(float));
    cudaMalloc(&device_varOcc[1], ( ((numObjs/OCC_MEANVAR_REDUCE_BLOCK_SIZE+1)/OCC_MEANVAR_REDUCE_BLOCK_SIZE+1) * numDims * MaxMixNum * sizeof(float)));


    meanOcc = (float *) malloc(MaxMixNum * numDims * sizeof(float));
    varOcc = (float *) malloc(MaxMixNum * numDims * sizeof(float));
    //    cudaMalloc(&device_meanStatistics, numObjs * numClusters * numDims * sizeof(float));
}

GMMTrain::~GMMTrain()
{
    if (NULL!=m_pfVarFloor) Free(m_pfVarFloor);

    m_pfVarFloor=NULL;

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

    Free(dOcc);

    if(device_reduceOcc[1]) {
        cudaFree(device_reduceOcc[1]);
        device_reduceOcc[1] = NULL;
    }

    Free(meanOcc);
    Free(varOcc);
    
    Free2D(probabilities);
    if(device_probabilities) {
        cudaFree(device_probabilities);
        device_probabilities = NULL;
    }
}


void GMMTrain::SetFloor(float p_fOccFloor,float p_fWeightFloor,float p_fVarFloor)
{
    ASSERT2(p_fVarFloor>0.f,"Error call SetFloor() : p_fVarFloor<0!");
    ASSERT2(p_fOccFloor>0.f,"Error call SetFloor() : p_fOccFloor<0!");
    ASSERT2(p_fWeightFloor>0.f,"Error call SetFloor() : p_fWeightFloor<0!");

    m_fOccFloor   = p_fOccFloor;
    m_fWeightFloor= p_fWeightFloor;
    m_fVarFloor   = p_fVarFloor;
}

void GMMTrain::UpdateModels()
{
    //printf("wengshtcocc 0 %f\n", dOcc[0]);
    //printf("wengshtcmean %f\n", meanOcc[0]);
    //printf("wengshtcvar %f\n",varOcc[0]);

    int	  vfloorNum,wfloorNum,nNoUpdateNum;
    float fminOcc,fmaxOcc;

    float		fWeightSum;	

    int i = 0;

    fWeightSum=0.0;

    vfloorNum=wfloorNum=nNoUpdateNum=0;

    fminOcc = fmaxOcc = dOcc[0]; 

    for(int m=0;m<m_nMixNum;m++)
    {
        if(dOcc[m] < fminOcc)
            fminOcc = dOcc[m];
        
        if(dOcc[m] > fmaxOcc)
            fmaxOcc = dOcc[m];

        if(dOcc[m] > m_fOccFloor)
        {
            m_pGmmModel[0].pfWeight[m] = dOcc[m] / totalOcc; 

            if(m_pGmmModel[0].pfWeight[m]<m_fWeightFloor)
            {
                m_pGmmModel[0].pfWeight[m]=m_fWeightFloor;
                wfloorNum++;
            }

            fWeightSum += m_pGmmModel[0].pfWeight[m];

            m_pGmmModel[0].pGauss[m].dMat=0.0;

            double occFact=1.0f/dOcc[m]; 
            for(int j=0;j<m_nVecSize;j++)
            {					
                int occIndex = m*numDims + j;
                int dimOccIndex = j * numClusters + m;
                
                m_pGmmModel[0].pGauss[m].pfMean[j] = meanOcc[occIndex] * occFact; 

                double dVar = varOcc[occIndex] * occFact
                    - m_pGmmModel[0].pGauss[m].pfMean[j]*m_pGmmModel[0].pGauss[m].pfMean[j];
                if(dVar<m_fVarFloor)
                {
                    m_pGmmModel[0].pGauss[m].pfDiagCov[j] = 1.f/m_fVarFloor;
                    vfloorNum++;
                }
                else
                    m_pGmmModel[0].pGauss[m].pfDiagCov[j] = 1.f/dVar;

                m_pGmmModel[0].pGauss[m].dMat += log(m_pGmmModel[0].pGauss[m].pfDiagCov[j]);
            }

            m_pGmmModel[0].pGauss[m].dMat -= m_nVecSize*log2pi;
            m_pGmmModel[0].pGauss[m].dMat *= 0.5;	
        }
        else
        {
            m_pGmmModel[0].pGauss[m].dMat -= log(m_pGmmModel[0].pfWeight[m]); 

            nNoUpdateNum++;
            fWeightSum += m_pGmmModel[0].pfWeight[m];
        }// end else

    }// end for(int m=0;m<m_nMixNum;m++)


    float ftmp=0.f;
    for(int m=0;m<m_nMixNum;m++)
    {
        m_pGmmModel[0].pfWeight[m] /= fWeightSum;  // 保证Weight之和等于1

        m_pGmmModel[0].pGauss[m].dMat += log(m_pGmmModel[0].pfWeight[m]);

        if (m_pGmmModel[0].pGauss[m].dMat>=0.0)
            printf("Warning : m_pGmmModel[%d].pGauss[%d].dMat=%.3f!\n",i,m,m_pGmmModel[0].pGauss[m].dMat);
    }

//    printf("============== update %d-th GMM-Model==========\n",i);
    //    printf("TotalProb=%12.6f, TotalFmNum=%5d, avgProb=%12.6f\n",
    //            m_pGMMStatic[0].fTotalProb,
    //            m_pGMMStatic[0].nTotalNum,
    //            m_pGMMStatic[0].fTotalProb/numObjs);

  //  printf("number of no update is %d\n",nNoUpdateNum);
  //  printf("%d floored weight, %d floored variance\n",wfloorNum,vfloorNum);
  //  printf("min mixture occ %f\nmax mixture occ %f\n",fminOcc,fmaxOcc);

}

void GMMTrain::EMIteration() {
    EMInitalize();
    
    cuda_ComputeStatistics();

    UpdateModels();
}



void GMMTrain::cuda_ComputeStatistics() {
    int numBlocks = (numObjs - 1 ) / OBSERVATE_BLOCK_SIZE + 1;
    int sharedSize = 2 * numDims * numClusters * sizeof(float) +  numClusters * sizeof(float);
    
    if(sharedSize > 49152) {
        sharedSize = 1;
    }

    cuda_observate_probability<<<numBlocks, OBSERVATE_BLOCK_SIZE, sharedSize>>>(sharedSize, device_dimObjects, device_dimClusters, device_diagCovs, device_dMats, numObjs, numDims, numClusters, device_probabilities);
    
    //cudaMemcpy(probabilities[0], device_probabilities, sizeof(float) * numObjs * numClusters, cudaMemcpyDeviceToHost);

    int sub_numObjs; 
    int rollIdx;
    int reduceSharedSize;

    /// reduce first-order 
    bool firstReduce = true;
    rollIdx = 1;
    numBlocks = numObjs;
    reduceSharedSize = OCC_MEANVAR_REDUCE_BLOCK_SIZE * numDims * sizeof(float) * 2; // 

    do {
        sub_numObjs = numBlocks;
        numBlocks = (numBlocks - 1) / OCC_MEANVAR_REDUCE_BLOCK_SIZE + 1;

        for(int clusterIdx = 0;clusterIdx < numClusters;clusterIdx++) {
            cuda_accumulate_meanVar_occ<<<numBlocks, OCC_MEANVAR_REDUCE_BLOCK_SIZE, reduceSharedSize>>>(firstReduce, device_dimObjects, device_probabilities, device_meanOcc[rollIdx], device_varOcc[rollIdx], sub_numObjs, numClusters, numDims, clusterIdx, device_meanOcc[rollIdx ^ 1], device_varOcc[rollIdx ^ 1]);
        }

        firstReduce = false;

        rollIdx ^= 1;
    } while(numBlocks != 1);

    cudaMemcpy(meanOcc, device_meanOcc[rollIdx], sizeof(float) * numClusters * numDims, cudaMemcpyDeviceToHost);
    cudaMemcpy(varOcc, device_varOcc[rollIdx], sizeof(float) * numClusters * numDims, cudaMemcpyDeviceToHost);
    //printf("xxd %f\n", meanOcc[0]);

    /// reduce zero-order
    numBlocks = numObjs;

    rollIdx = 0;

    int clusterPerCUDA = min(4, numClusters);
    reduceSharedSize = OCC_REDUCE_BLOCK_SIZE * sizeof(float) * clusterPerCUDA; //numClusters; 1024*4*4 = 16k
    do {
        sub_numObjs = numBlocks;
        numBlocks = (numBlocks - 1) / OCC_REDUCE_BLOCK_SIZE + 1;

        // device_reduceOcc[0] == device_probabilities
        for(int clusterStart = 0; clusterStart < numClusters; clusterStart += clusterPerCUDA)
            cuda_accumulate_occ<<<numBlocks, OCC_REDUCE_BLOCK_SIZE, reduceSharedSize>>>(device_reduceOcc[rollIdx], sub_numObjs, numClusters, clusterStart, clusterPerCUDA, device_reduceOcc[rollIdx ^ 1]);

        rollIdx ^= 1;
    } while(numBlocks != 1);

    cudaMemcpy(dOcc, device_reduceOcc[rollIdx], sizeof(float) * numClusters, cudaMemcpyDeviceToHost);

    totalOcc = 0;
    for(int m = 0;m < numClusters; m++) {
        totalOcc += dOcc[m];
    }
}

void GMMTrain::EMInitalize() {
    // Initalize all the means and diagcovs for mixture models
    numClusters = m_nMixNum;
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
}

void GMMTrain::EMTrain(int EMIterNum) {
#ifdef PERFORMANCE_REPORT
    float avg = 0;
    int   cnt = 0;
    double startT = 0, endT = 0;
#endif

    for(int iEMIndex=1 ;iEMIndex<=EMIterNum; iEMIndex++) {
#ifdef PERFORMANCE_REPORT
        startT = wtime();
#endif
        EMIteration();
#ifdef PERFORMANCE_REPORT
        endT   = wtime();
        avg += endT - startT;
        cnt ++;
#endif
    }
#ifdef PERFORMANCE_REPORT
    printf("Avg EM Iteration: %f\n", avg / cnt);
#endif
}

void GMMTrain::AllocProbBuf(int m_nMaxFrameNum, int m_nMixNum)
{}

void GMMTrain::AllocStaticBuf(int m_nModelNum, int m_nMixNum, int m_nVecSize4, int m_nMaxFrameNum)
{}

void GMMTrain::FreeStaticBuf()
{}

void GMMTrain::ResetStaticBuf()
{}

void GMMTrain::ComputeStatiscs_MT(float *m_pfFeatBuf, int nFrameNum, int p_nModelIdx) 
{}
