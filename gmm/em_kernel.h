#ifndef GMM_TERMPROJ_EM_KERNEL_H
#define GMM_TERMPROJ_EM_KERNEL_H

#define OBSERVATE_BLOCK_SIZE 1024
#define OCC_MEANVAR_REDUCE_BLOCK_SIZE 128
#define OCC_REDUCE_BLOCK_SIZE 1024

#define LOG_INF 1e120

#include "comm_srlr.h"


__global__ void cuda_observate_probability(int sharedSize, float * dimObjects, float *means, float *diagCovs, float *dMat, int numObjs, int numDims, int numClusters, float *probabilities);

__global__ void cuda_accumulate_occ(float * device_mapOcc, int numObjs, int numClusters, int clusterStart, int sub_numClusters, float *device_reduceOcc);


__global__ void cuda_accumulate_meanVar_occ(bool firstReduce, float *device_dimObjects, float * device_probabilities, float * device_mapMeanOcc, float *device_mapVarOcc, int sub_numObjs, int numClusters, int numDims, int clusterIdx, float * device_reduceMeanOcc, float *device_reduceVarOcc);
        
__device__ float cuda_logAdd(float a, float b);

#endif
