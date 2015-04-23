// =====================================================================================
// 
//       Filename:  test_model.cpp
// 
//    Description:  This program is written to check if the cuda-version give correct model-training process
// 
//        Version:  0.01
//        Created:  04/21/2015 05:31:19 PM
//       Revision:  none
//       Compiler:  clang 3.5
// 
//         Author:  wengsht (SYSU-CMU), wengsht.sysu@gmail.com
//        Company:  
// 
// =====================================================================================

#include "GMMParam.h"

void report(GMMParam &gmmParam);

int main(int argc, char **argv) {
    if(argc < 3) {
        printf("usage: ./test_model CPU_TRAINED_MODEL CUDA_TRAINED_MODEL");
    }
    GMMParam cpu_trained_model, cuda_trained_model;
    
    cpu_trained_model.LoadModel(argv[1]);
    
    cuda_trained_model.LoadModel(argv[2]);
    
    report(cpu_trained_model);
    report(cuda_trained_model);
    
    return 0;
}

void report(GMMParam &gmmParam) {
    gmmParam.report();
}
