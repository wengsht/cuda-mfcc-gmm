#include <stdio.h>
#include <stdlib.h>


__global__ void cuda_find_nearest_cluster(int sharedSize, float * dimObjects, float *clusters, int numObjs, int numDims, int numClusters, int *memberShip);

__global__ void cuda_accumulate_clusters(float * dimObjects, int *memberShip, int numObjs, int numDims, int numClusters, int * clusterSize, float *clusters);

__global__ void cuda_average_clusters(int * clusterSize, float * clusters);

__global__ void cuda_accumulate_diagcovs(float *device_dimObjects, int * device_memberShip, float * device_dimClusters, int numObjs, int numDims, int numClusters, float * device_diagCovs);

__global__ void cuda_average_diagcovs(int * device_clusterSize, float * device_diagCovs, float * device_dMats);

inline __device__ float calculateDist(float * dimObjects, float * shared_clusters, int numObjs, int numClusters, int numDims, int ObjsIdx, int clusterIdx);


#define FIND_NEAREST_BLOCKSIZE 1024

#define ACCUMULATE_BLOCKSIZE 1024
