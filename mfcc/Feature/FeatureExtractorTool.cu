
#include "FeatureExtractorTool.h"

//Y__global__ 
//Yvoid matrix_mul_kernel(d_type *sq_matrix_1, d_type *sq_matrix_2, d_type *sq_matrix_result, int dim_a, int dim_b, int dim_c) 
//Y{
//Y    int row = by * dy + ty;
//Y    int col[COL_STEP]; 
//Y
//Y    col[0] = COL_STEP * bx * dx + tx;
//Y    for(int i = 1;i < COL_STEP;i++)
//Y        col[i] = col[i-1] + BLOCK_SIZE;
//Y
//Y    for(int i  =0;i < COL_STEP;i++) 
//Y        if(row < dim_a && col[i] < dim_c)
//Y            sq_matrix_result[row * dim_c + col[i]] = 0;
//Y    
//Y    for(int k = 0;k < dim_b;k++) {
//Y        for(int i  =0;i < COL_STEP;i++) 
//Y            if(row < dim_a && col[i] < dim_c)
//Y                sq_matrix_result[row * dim_c + col[i]] += sq_matrix_1[row*dim_b + k] * sq_matrix_2[col[i] * dim_b + k];
//Y    }
//Y}

__global__
void mel2dct_kernel(FEATURE_DATA *d_melLogSpec_data, int unitSize, int cepsNum, double arg_PI){
    extern __shared__ FEATURE_DATA s_data[];
    
    size_t blockOffset = blockDim.x*blockIdx.x;
    size_t totalOffset = blockOffset+threadIdx.x;
    s_data[threadIdx.x] = d_melLogSpec_data[totalOffset]; 
    __syncthreads();
    
    int frameIdx = threadIdx.x/unitSize;
    int innerIdx = threadIdx.x % unitSize;
    
    if(innerIdx < cepsNum){
        int frameBegin = unitSize*frameIdx, 
            frameEnd = unitSize*(frameIdx+1);
    
        FEATURE_DATA result = 0;
        double constVal = arg_PI*innerIdx/unitSize;
        for(int j=frameBegin; j<frameEnd; j++){
            result += s_data[j]*cos(constVal*(j-frameBegin+0.5)); 
        } 

        d_melLogSpec_data[totalOffset] = result * sqrt(1.0/unitSize);
    }
}



__global__ 
void matrix_mul_kernel(d_type *sq_matrix_1, d_type *sq_matrix_2, d_type *sq_matrix_result, int dim_a, int dim_b, int dim_c) 
{
    //A cuda thread will calculate 4 results, result[row][col[0~4]]
    int row = by * dy + ty;
    int col[COL_STEP]; 
    int col_mul_dimb[COL_STEP]; 

    col[0] = COL_STEP * bx * dx + tx;
    col[1] = col[0] + BLOCK_SIZE;
    col[2] = col[1] + BLOCK_SIZE;
    col[3] = col[2] + BLOCK_SIZE;
    
    col_mul_dimb[0] = col[0] * dim_b;
    col_mul_dimb[1] = col[1] * dim_b;
    col_mul_dimb[2] = col[2] * dim_b;
    col_mul_dimb[3] = col[3] * dim_b;

    //One shared copy of sq_matrix_1 can be used to calculate 4 blocks,
    //increase the utilization of share memory
    __shared__ d_type s_a[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ d_type s_b[BLOCK_SIZE][BLOCK_SIZE][COL_STEP];

    d_type val[COL_STEP] = {0.0, 0.0, 0.0, 0.0};

    int rowIdx = row * dim_c;
    int rowAIdx = row * dim_b; 

    //pre calculate index of matrixes. 
    int sq_matrix_1_index = rowAIdx + tx;
    int sq_matrix_2_index = (ty) ; 
    int sq_matrix_1_step = BLOCK_SIZE;
    int sq_matrix_2_step = BLOCK_SIZE ;

    int ks, k;

    //pre fetch values into local memory. 
    d_type preFetchA;
    d_type * sb_sq_matrix_p = &(s_b[ty][tx][0]);

    int K_STEP = COL_STEP * BLOCK_SIZE;
    int ceil_ks;

    d_type *s_b_sq_matrix_used;

     //ks is BLOCK_SIZE step length, and one iteration will calculate 4 block
    for(ks = 0; ks <= dim_b-BLOCK_SIZE; ks += BLOCK_SIZE, sq_matrix_1_index += sq_matrix_1_step, sq_matrix_2_index += sq_matrix_2_step) {

        //fetch matrix1 and matrix2 into shared memory
        s_a[ty][tx] = sq_matrix_1[sq_matrix_1_index];

        sb_sq_matrix_p[0] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[0]];
        sb_sq_matrix_p[1] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[1]];
        sb_sq_matrix_p[2] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[2]];
        sb_sq_matrix_p[3] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[3]];

        __syncthreads();

        s_b_sq_matrix_used = &(s_b[0][tx][0]);

        for(k = 0; k < BLOCK_SIZE; k++) {
            preFetchA = s_a[ty][k]; 
            val[0] += preFetchA * s_b_sq_matrix_used[0];  
            val[1] += preFetchA * s_b_sq_matrix_used[1]; 
            val[2] += preFetchA * s_b_sq_matrix_used[2];
            val[3] += preFetchA * s_b_sq_matrix_used[3];
            s_b_sq_matrix_used += K_STEP; 
        }

        __syncthreads();
    }

    //because the dimension is not always power of 2, we need to add a tail for the rest calculation
    if(ks < dim_b) {
        if(col[0] < dim_c && row < dim_a)
            s_a[ty][tx] = sq_matrix_1[sq_matrix_1_index];

        if(col[0] < dim_c && row < dim_a) 
            sb_sq_matrix_p[0] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[0]];
        if(col[1] < dim_c && row < dim_a) 
            sb_sq_matrix_p[1] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[1]];
        if(col[2] < dim_c && row < dim_a) 
            sb_sq_matrix_p[2] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[2]];
        if(col[3] < dim_c && row < dim_a) 
            sb_sq_matrix_p[3] = sq_matrix_2[sq_matrix_2_index + col_mul_dimb[3]];

        __syncthreads();
        s_b_sq_matrix_used = &(s_b[0][tx][0]) - K_STEP;

        ceil_ks = dim_b-ks;

#pragma unroll 32
        for(k=0; k < ceil_ks; k++) {
            preFetchA = s_a[ty][k]; 
            s_b_sq_matrix_used += K_STEP; 
            val[0] += preFetchA * s_b_sq_matrix_used[0];  
            val[1] += preFetchA * s_b_sq_matrix_used[1]; 
            val[2] += preFetchA * s_b_sq_matrix_used[2];
            val[3] += preFetchA * s_b_sq_matrix_used[3];
        }
        __syncthreads();
    }

    // Write the results back to global memory
    if(row >= dim_a) return;

    //if(col[0] < dim_c)
    //    sq_matrix_result[rowIdx + col[0]] = val[0];
    //if(col[1] < dim_c)
    //    sq_matrix_result[rowIdx + col[1]] = val[1];
    //if(col[2] < dim_c)
    //    sq_matrix_result[rowIdx + col[2]] = val[2];
    //if(col[3] < dim_c)
    //    sq_matrix_result[rowIdx + col[3]] = val[3];
    

    if(col[0] < dim_c){
        sq_matrix_result[rowIdx + col[0]] = log(0.0001+fabs(val[0]));
    }
    if(col[1] < dim_c){
        sq_matrix_result[rowIdx + col[1]] = log(0.0001+fabs(val[1]));
    }
    if(col[2] < dim_c){
        sq_matrix_result[rowIdx + col[2]] = log(0.0001+fabs(val[2]));
    }
    if(col[3] < dim_c){
        sq_matrix_result[rowIdx + col[3]] = log(0.0001+fabs(val[3]));
    }

}


__global__
void windowFFT_kernel(FEATURE_DATA *d_SpeechSignal_real, FEATURE_DATA *d_SpeechSignal_imag, int frameNum, int frameSize, int f, int selIdx, double arg) {
    extern __shared__ char s_SpeechSignal[];
    int p, i, j, rollIdx=0, oldRollIdx;
    size_t innerIdx = threadIdx.x % frameSize, 
           frame_offset = blockDim.x*blockIdx.x+(threadIdx.x/frameSize)*frameSize;
    
    FEATURE_DATA temp_cp[2], temp_wm[2], temp_w[2], temp_result[2];
    size_t total_offset = frame_offset+innerIdx;
   
    //cp *temp = (cp *) temp_cp, 
    //   *wm = (cp*)temp_wm, 
    //   *w = (cp*)temp_w; 
    
    FEATURE_DATA *s_signal_real[2]; 
    FEATURE_DATA *s_signal_imag[2]; 

    size_t sharedSize = blockDim.x * sizeof(FEATURE_DATA);
    s_signal_real[0] = (FEATURE_DATA *) s_SpeechSignal;
    s_signal_imag[0] = (FEATURE_DATA *) &s_SpeechSignal[sharedSize];
    s_signal_real[1] = (FEATURE_DATA *) &s_SpeechSignal[2*sharedSize];
    s_signal_imag[1] = (FEATURE_DATA *) &s_SpeechSignal[3*sharedSize];

    *(s_signal_real[0]+innerIdx) = *(d_SpeechSignal_real+total_offset);
    *(s_signal_imag[0]+innerIdx) = *(d_SpeechSignal_imag+total_offset);
    __syncthreads();

    int tmpIdx;
    for(int k = frameSize>>1; k; k>>=1, arg*=0.5) {
        rollIdx ^= 1;
        oldRollIdx = rollIdx^1;

        getPolarValue(1, f*arg, temp_wm);
        temp_w[0] = 1;
        temp_w[1] = 0;

        i = innerIdx/k;
        j = innerIdx%k;
        for(int t=0; t<i; t++){
            //w = w*wm;
            mulComplex(temp_w, temp_wm, temp_w);
        }
        i = i*k;
        p = i<<1;
        if(p>=frameSize) p-=frameSize;

        //mulComplex(temp, w, s_signal[oldRollIdx]+(p+k+j)); 
        tmpIdx = p+k+j;
        temp_result[0] = s_signal_real[oldRollIdx][tmpIdx];
        temp_result[1] = s_signal_imag[oldRollIdx][tmpIdx];
        mulComplex(temp_cp, temp_w, temp_result);
        
        //addComplex(s_signal[rollIdx]+(i+j), temp, s_signal[oldRollIdx]+(p+j));
        tmpIdx = p+j;
        temp_result[0] = s_signal_real[oldRollIdx][tmpIdx];
        temp_result[1] = s_signal_imag[oldRollIdx][tmpIdx];
        addComplex(temp_result, temp_cp, temp_result); 

        tmpIdx = i+j;
        s_signal_real[rollIdx][tmpIdx] = temp_result[0];
        s_signal_imag[rollIdx][tmpIdx] = temp_result[1];

        __syncthreads();
    }
    
    //d_SpeechSignal[frame_offset+innerIdx] = *(s_signal[selIdx]+innerIdx);
    tmpIdx = frame_offset+innerIdx;
    d_SpeechSignal_real[tmpIdx] = s_signal_real[selIdx][innerIdx];
    d_SpeechSignal_imag[tmpIdx] = s_signal_imag[selIdx][innerIdx];
}


__global__
void preProcessing_kernel(SOUND_DATA *d_rd, int rd_size, FEATURE_DATA *d_window_data, int samplePerWin, int stepPerWin, double factor, double arg_PI_factor){
    if(threadIdx.x<samplePerWin){ 
        //int frameIdx = blockIdx.x;
        //int innerIdx = threadIdx.x;
        size_t rd_idx = blockIdx.x*stepPerWin + threadIdx.x;
        
        if(rd_idx>=rd_size)
            return;

        size_t final_idx = blockIdx.x*blockDim.x + threadIdx.x;
    
        FEATURE_DATA result;

        if(rd_idx==0)
            result = 0;
        else
            result = 1.0*d_rd[rd_idx] - factor*d_rd[rd_idx-1];
   
        result *= (0.5-0.5*cos(arg_PI_factor*threadIdx.x));
    
        d_window_data[final_idx] = result;
    }
}



__device__ 
void mulComplex(FEATURE_DATA *output, FEATURE_DATA *input1, FEATURE_DATA *input2){
    FEATURE_DATA real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    output[0] = real1*real2-imag1*imag2;
    output[1] = real1*imag2+imag1*real2;
    //output = cp( real1*real2-imag1*imag2 , real1*imag2+imag1*real2 );
}

__device__
void addComplex(FEATURE_DATA *output, FEATURE_DATA *input1, FEATURE_DATA *input2){
    FEATURE_DATA real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    output[0] = real1+real2;
    output[1] = imag1+imag2;
    //output = cp( real1+real2, imag1+imag2 );
}

__device__
void getRealImag(FEATURE_DATA& real, FEATURE_DATA& imag, const FEATURE_DATA *input){
    real = input[0];
    imag = input[1];
}

__device__
void getPolarValue(FEATURE_DATA rho, FEATURE_DATA theta, FEATURE_DATA *output){
    *output = rho*cos(theta);
    *(output+1) = rho*sin(theta);
}


__device__ 
void mulComplex(cp *output, cp *input1, cp *input2){
    double real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    double *ptr_output = (double *)output;
    *ptr_output = real1*real2-imag1*imag2;
    *(ptr_output+1) = real1*imag2+imag1*real2;
    //output = cp( real1*real2-imag1*imag2 , real1*imag2+imag1*real2 );
}

__device__
void addComplex(cp *output, cp *input1, cp *input2){
    double real1, imag1, real2, imag2;
    getRealImag(real1,imag1,input1);
    getRealImag(real2,imag2,input2);
    double *ptr_output = (double *)output;
    *ptr_output = real1+real2;
    *(ptr_output+1) = imag1+imag2;
    //output = cp( real1+real2, imag1+imag2 );
}

__device__
void getRealImag(double& real, double& imag, const cp *input){
    double *comp = (double *)input;
    real = *comp;
    imag = *(comp+1);
}

//__device__
//void getPolarValue(double rho, double theta, double *output){
//    *output = rho*cos(theta);
//    *(output+1) = rho*sin(theta);
//}

