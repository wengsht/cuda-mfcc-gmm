#include "em_kernel.h"

__global__ void cuda_observate_probability(int sharedSize, float * dimObjects, float *means, float *diagCovs, float *dMat, int numObjs, int numDims, int numClusters, float *probabilities) {
    extern __shared__ char sharedMemory[];

    float * shared_means = (float *)sharedMemory;
    float * shared_diagCovs = (shared_means + numClusters * numDims);
    float * shared_dMats = (shared_diagCovs + numClusters * numDims);

    int objIndex = blockDim.x * blockIdx.x + threadIdx.x;
    
    /// not using shared memory
    if(sharedSize == 1) {
        shared_means = means;
        shared_dMats = dMat;
        shared_diagCovs = diagCovs;
    }
    else {
        for(int i = threadIdx.x; i < numClusters; i+= blockDim.x) {
            shared_dMats[i] = dMat[i];
            for(int j = 0;j < numDims; j++) {
                int mIdx = j * numClusters + i;
                shared_means[mIdx] = means[mIdx];
                shared_diagCovs[mIdx] = diagCovs[mIdx];
            }
        }
    }

    __syncthreads();

    float pdf, val, sigma;

    if(objIndex < numObjs) {
        sigma = - LOG_INF;

        for(int clusterIdx = 0; clusterIdx < numClusters; clusterIdx ++) {
            pdf = 0.0;
            for(int i = 0;i < numDims; i++) {
                val = (dimObjects[i*numObjs + objIndex] - shared_means[i*numClusters + clusterIdx]);
                pdf -=  0.5 * val * val * shared_diagCovs[i * numClusters + clusterIdx];
            }
            pdf += shared_dMats[clusterIdx];
            probabilities[objIndex * numClusters + clusterIdx] = pdf;

            sigma = cuda_logAdd(sigma, pdf);
        }

        for(int clusterIdx = 0; clusterIdx < numClusters; clusterIdx ++) {
            pdf = probabilities[objIndex * numClusters + clusterIdx];
            pdf -= sigma;
            pdf = powf(e, pdf);
            probabilities[objIndex * numClusters + clusterIdx] = pdf;
        }
    }
}

__global__ void cuda_accumulate_occ(float * device_mapOcc, int numObjs, int numClusters, int clusterStart, int sub_numClusters,  float *device_reduceOcc) {
    int objIndex = blockDim.x * blockIdx.x + threadIdx.x;
    extern __shared__ float shared_objects[];

    if(objIndex < numObjs) {
        for(int i = 0;i < sub_numClusters; i++) 
            //        for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--) 
            shared_objects[threadIdx.x * sub_numClusters + i] = device_mapOcc[objIndex * numClusters + i + clusterStart];
    }
    else {
        for(int i = 0;i < sub_numClusters; i++)
            //for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--) 
            shared_objects[threadIdx.x * sub_numClusters + i] = 0;
    }

    __syncthreads();

    for(int i = (blockDim.x >> 1); i >= 1; i>>=1) {
        if(threadIdx.x < i) {
            for(int j = 0;j < sub_numClusters; j++) {
                //for(int j = clusterStart + sub_numClusters - 1;j >= clusterStart; j--) 
                shared_objects[threadIdx.x * sub_numClusters + j] += shared_objects[(threadIdx.x + i) * sub_numClusters + j];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        for(int i = 0;i < sub_numClusters;i++) {
            //        for(int i = clusterStart + sub_numClusters - 1;i >= clusterStart; i--) 
            device_reduceOcc[blockIdx.x * numClusters + i + clusterStart]  = shared_objects[i];
        }
    }
}

#define getIndex3(a,b,c,d,e) ((a)*(d)*(e) + (b)*(e) + (c));
#define getIndex2(a,b,c) ((a)*(c) + (b))

__global__ void cuda_accumulate_meanVar_occ(bool firstReduce, float *device_dimObjects, float * device_probabilities, float * device_mapMeanOcc, float *device_mapVarOcc, int numObjs, int numClusters, int numDims, int clusterIdx, float * device_reduceMeanOcc, float *device_reduceVarOcc) {
    extern __shared__ float occ_sharedMemory[];

    int objIndex = blockDim.x * blockIdx.x + threadIdx.x;

    float * shared_meanOcc = occ_sharedMemory;
    float * shared_varOcc  = shared_meanOcc + numDims * blockDim.x;

    float feature;
    float probability;

    if(objIndex < numObjs) {
        if(firstReduce) {
            probability = device_probabilities[objIndex * numClusters + clusterIdx];

            for(int dimIdx = 0; dimIdx < numDims; dimIdx++) {
                int sharedIndex = getIndex2(threadIdx.x, dimIdx, numDims);
                feature = device_dimObjects[(dimIdx) * numObjs + objIndex];

                shared_meanOcc[sharedIndex] = probability * feature;

                shared_varOcc[sharedIndex] = feature * shared_meanOcc[sharedIndex];
            }
        }
        else {
            for(int dimIdx = 0; dimIdx < numDims; dimIdx++) {
                //int sharedIndex = getIndex3(threadIdx.x, clusterIdx, dimIdx, numClusters, numDims);
                int sharedIndex = getIndex2(threadIdx.x, dimIdx, numDims);
                int globalIndex = getIndex3(objIndex, clusterIdx, dimIdx , numClusters, numDims);
                shared_meanOcc[sharedIndex] = device_mapMeanOcc[globalIndex];
                shared_varOcc[sharedIndex] = device_mapVarOcc[globalIndex];
            }
        }
    }
    else {
        for(int dimIdx = 0; dimIdx < numDims; dimIdx++) {
            //int sharedIndex = getIndex3(threadIdx.x, clusterIdx, dimIdx, numClusters, numDims);
            int sharedIndex = getIndex2(threadIdx.x, dimIdx, numDims);
            shared_meanOcc[sharedIndex] = 0;
            shared_varOcc[sharedIndex] = 0;
        }
    }

    __syncthreads();

    for(int i = (blockDim.x >> 1); i >= 1; i >>= 1) {
        if(threadIdx.x < i) {
            for(int dimIdx = 0; dimIdx < numDims; dimIdx ++) {
                int toIndex = getIndex2(threadIdx.x, dimIdx, numDims);
                int fromIndex = getIndex2(threadIdx.x + i, dimIdx, numDims);

                shared_meanOcc[toIndex] += shared_meanOcc[fromIndex];
                shared_varOcc[toIndex] += shared_varOcc[fromIndex];
            }
        }
        __syncthreads();
    }

    if(threadIdx.x == 0) {
        for(int dimIdx = 0; dimIdx < numDims; dimIdx ++) {
            int sharedIndex = getIndex2(threadIdx.x, dimIdx, numDims);
            int globalIndex = getIndex3(blockIdx.x, clusterIdx, dimIdx, numClusters, numDims);

            device_reduceMeanOcc[globalIndex] = shared_meanOcc[sharedIndex];
            device_reduceVarOcc[globalIndex] = shared_varOcc[sharedIndex];
        }
    }
}

__device__ float cuda_logAdd(float loga, float logb) {
    float ret = 0.0;
    if(loga > logb) {
        ret = loga;
        loga = logb;
        logb = ret;
    }
    ret  = logb + log(1.0 + powf(e, loga-logb));
    return ret;
}
