set(FEATURE Feature)
set(MFCC_CUDA MfccCuda)

add_library(${FEATURE} Feature.cpp FeatureExtractor.cpp mathtool.cpp wtime.cpp)


cuda_add_library(${MFCC_CUDA} Feature.cpp FeatureExtractorTool.cu FeatureExtractor.cu mathtool.cpp wtime.cpp)
