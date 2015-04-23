#include "kmean_kernel.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void cuda_find_nearest_cluster(int sharedSize, float * dimObjects, float *clusters, int numObjs, int numDims, int numClusters, int *memberShip) {
    extern __shared__ float  shareMemory[];
    float *shared_clusters = shareMemory;
    
    if(sharedSize == 1) {
        shared_clusters = clusters;
    }
    else {
        /// Copy Clusters into shared memory
        for(int idx = threadIdx.x; idx < numClusters; idx += blockDim.x) {
            for(int idy = 0; idy < numDims; idy ++) {
                shared_clusters[idy * numClusters + idx] = clusters[idy * numClusters + idx];
            }
        }
    }

    __syncthreads();

    int ObjsIdx = blockDim.x * blockIdx.x + threadIdx.x;

    if(ObjsIdx < numObjs) {
        int index = 0;

        float dist, minDist;

        minDist = calculateDist(dimObjects, shared_clusters, numObjs, numClusters, numDims, ObjsIdx, 0);

        for(int clusterIdx = 1; clusterIdx < numClusters; clusterIdx ++) {
            dist = calculateDist(dimObjects, shared_clusters, numObjs, numClusters, numDims, ObjsIdx, clusterIdx);

            if(minDist > dist) {
                index = clusterIdx;

                minDist = dist;
            }
        }
        memberShip[ObjsIdx] = index;
    }
}

inline __device__ float calculateDist(float * dimObjects, float * shared_clusters, int numObjs, int numClusters, int numDims, int ObjsIdx, int clusterIdx) {
    float res = 0.0, tmp;

    for(int i = 0;i < numDims; i++) {
        tmp = dimObjects[i * numObjs + ObjsIdx] - shared_clusters[i * numClusters + clusterIdx];

        res += tmp * tmp;
    }

    return (res);
}

__global__ void cuda_accumulate_clusters(float * dimObjects, int *memberShip, int numObjs, int numDims, int numClusters, int *clusterSize, float *clusters) {
    int ObjsIdx = blockDim.x * blockIdx.x + threadIdx.x;
    float val;

    if(ObjsIdx < numObjs) {
        int index = memberShip[ObjsIdx];

        atomicAdd(&clusterSize[index], 1);
        for(int idx = 0; idx < numDims; idx ++) {
            val = dimObjects[idx * numObjs + ObjsIdx];
            atomicAdd(&(clusters[idx * numClusters + index]), val);
        }
    }
}

__global__ void cuda_average_clusters(int * clusterSize, float * clusters) {
    clusters[blockDim.x * blockIdx.x + threadIdx.x] /= clusterSize[threadIdx.x];
}

__global__ void cuda_average_diagcovs(int * device_clusterSize, float * device_diagCovs, float * device_dMats) {
    int clusterIdx = threadIdx.x, dimIdx = blockIdx.x, numClusters = blockDim.x;

    int idx = numClusters * dimIdx + clusterIdx;
    float val = device_diagCovs[idx] / device_clusterSize[clusterIdx];

    val = 1.0 / val;
    device_diagCovs[idx] = val;

    atomicAdd(&device_dMats[clusterIdx], log(val));
}

__global__ void cuda_accumulate_diagcovs(float *device_dimObjects, int * device_memberShip, float * device_dimClusters, int numObjs, int numDims, int numClusters, float * device_diagCovs) {
    int ObjsIdx = blockDim.x * blockIdx.x + threadIdx.x;

    float val;
    int idy;
    if(ObjsIdx < numObjs) {
        int index = device_memberShip[ObjsIdx];

        for(int idx = 0; idx < numDims; idx ++) {
            idy = idx * numClusters + index;
            val = device_dimObjects[idx * numObjs + ObjsIdx] - device_dimClusters[idy];

            atomicAdd(& device_diagCovs[idy], val * val);
        }
    }
}

