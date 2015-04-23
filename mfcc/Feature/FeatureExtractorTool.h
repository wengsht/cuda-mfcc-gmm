#pragma once
#ifndef __SpeechRecognitionSystem__FeatureExtractionTool__
#define __SpeechRecognitionSystem__FeatureExtractionTool__

#include <iostream>
#include <complex>
#include "mathtool.h"

#define d_type double
#define BLOCK_SIZE 16

#define K_UNROLL_STEP 32 // [1,2,4,8,16]

#if K_UNROLL_STEP > BLOCK_SIZE
#undef K_UNROLL_STEP
#define K_UNROLL_STEP BLOCK_SIZE
#endif

#define COL_STEP 4

#define ty (threadIdx.y)
#define tx (threadIdx.x)

#define by (blockIdx.y)
#define bx (blockIdx.x)

#define dy (blockDim.y)
#define dx (blockDim.x)

//#define e 2.718281828459
//typedef std::complex<double> cp;
//#ifndef __SpeechRecongnitionSystem__PI__
//#define __SpeechRecongnitionSystem__PI__
//const double PI = std::acos(-1);
//#endif

__global__
void mel2dct_kernel(FEATURE_DATA *d_melLogSpec_data, int unitSize, int cepsNum, double arg_PI = PI);

__global__ 
void matrix_mul_kernel(d_type *sq_matrix_1, d_type *sq_matrix_2, d_type *sq_matrix_result, int dim_a, int dim_b, int dim_c);
    
__global__
void windowFFT_kernel(FEATURE_DATA *d_SpeechSignal_real, FEATURE_DATA *d_SpeechSignal_imag, int frameNum, int frameSize, int f, int selIdx, double arg=PI);

__global__
void preProcessing_kernel(SOUND_DATA *d_rd, int rd_size, FEATURE_DATA *d_window_data, int samplePerWin, int stepPerWin, double factor, double arg_PI_factor);

__device__ 
void mulComplex(FEATURE_DATA *output, FEATURE_DATA *input1, FEATURE_DATA *input2);

__device__ 
void addComplex(FEATURE_DATA *output, FEATURE_DATA *input1, FEATURE_DATA *input2);

__device__
void getRealImag(FEATURE_DATA& real, FEATURE_DATA& imag, const FEATURE_DATA *input);

__device__
void getPolarValue(FEATURE_DATA length, FEATURE_DATA angle, FEATURE_DATA* output);



__device__ 
void mulComplex(cp *output, cp *input1, cp *input2);

__device__ 
void addComplex(cp *output, cp *input1, cp *input2);

__device__
void getRealImag(double& real, double& imag, const cp *input);

//__device__
//void getPolarValue(double length, double angle, double* output);

#endif
