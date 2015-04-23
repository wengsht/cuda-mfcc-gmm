project (term_proj)
cmake_minimum_required (VERSION 2.8)

find_package(CUDA)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11 -gencode arch=compute_30,code=sm_30 -O3)

add_subdirectory("${PROJECT_SOURCE_DIR}/mfcc")
add_subdirectory("${PROJECT_SOURCE_DIR}/gmm")
