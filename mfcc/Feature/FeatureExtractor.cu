#include "FeatureExtractor.h"
#include "RawData.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "wtime.h"
#include "mathtool.h"
#include "ThreadPool.h"
#include "FeatureExtractorTool.h"

SP_RESULT FeatureExtractor::exFeatures(const RawData *data) {
    return exFeatures(data, \
            sampleRate,
            preEmpFactor, \
            winTime, \
            stepTime, \
            winFunc, \
            minF, \
            maxF, \
            hz2melFunc, \
            mel2hzFunc, \
            nfilts, \
            cepsNum);
}

SP_RESULT FeatureExtractor::exDoubleDeltaFeatures(const RawData *data) {
    exFeatures(data);

    doubleDelta(normalMelCeps);
    
    return SP_SUCCESS;
}
void FeatureExtractor::doubleDelta(std::vector<Feature> & normalMelCeps) {
    int idx, siz = normalMelCeps.size();

    for(idx = 0; idx < siz; idx ++) 
        normalMelCeps[idx].fillDoubleDelta();
}

SP_RESULT FeatureExtractor::exFeatures(const RawData *data, \
        int sampleRate, \
        double preEmpFactor, \
        double winTime, \
        double stepTime, \
        double (*winFunc)(int, int), \
        double minF, \
        double maxF, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double), \
        int nfilts, \
        int cepsNum) {
    //SP_RESULT res; 
    inital();

    double startT, finishT, initializeTime;
    double totalTime = 0;

    //startT = wtime();
    //preEmph(e_emp_data, data->getData(), data->getFrameNum(), preEmpFactor);
    //finishT = wtime();
    //double t_preemp = finishT-startT;
    //totalTime += t_preemp;

    //startT = wtime();
    //windowing(e_windows, e_emp_data, winTime, stepTime, sampleRate, winFunc);
    //finishT = wtime();
    //double t_window = finishT-startT;
    //totalTime += t_window;
    
    startT = wtime();
    initializeTime = preProcessing(e_windows, data->getData(), data->getFrameNum(), preEmpFactor, winTime, stepTime, sampleRate);
    finishT = wtime();
    double t_preProcessing = finishT-startT-initializeTime;
    totalTime += t_preProcessing;

    startT = wtime();
    powSpectrum(e_powSpec, e_windows);
    finishT = wtime();
    double t_powSpec = finishT-startT;
    totalTime += t_powSpec;
    

    int nfft = (e_powFrameSize -1) << 1;

    startT = wtime();
    fft2MelLog(nfft, &e_melLogSpec, e_powSpec, nfilts, hz2melFunc, mel2hzFunc, minF, maxF, sampleRate);
    finishT = wtime();
    double t_mel = finishT-startT;
    totalTime += t_mel;
    
    startT = wtime();
    melCepstrum(melCeps, e_melLogSpec, cepsNum);
    finishT = wtime();
    double t_dctCep = finishT-startT;
    totalTime += t_dctCep;

    startT = wtime();
    time_t start = time(0);
    normalization(normalMelCeps, melCeps);
    finishT = wtime();
    double t_norm = finishT-startT;
    totalTime += t_norm;

    doubleDelta(normalMelCeps);

    std::cout << "CUDA Initialize Time: " << initializeTime << std::endl;
    std::cout << "Total Time (Without InitializeTime) : " << totalTime << std::endl;
    //std::cout << "PreEmp: " << t_preemp << " s , " << t_preemp*100/totalTime <<"%" <<std::endl;
    //std::cout << "Windowing: " << t_window << " s , " << t_window*100/totalTime <<"%" << std::endl;
    std::cout << "PreProcessing: " << t_preProcessing << " s , " << t_preProcessing*100/totalTime <<"%"<< std::endl;
    std::cout << "FFT: " << t_powSpec << " s , " << t_powSpec*100/totalTime <<"%"<< std::endl;
    std::cout << "MelFiltering: " << t_mel << " s , " << t_mel*100/totalTime <<"%"<< std::endl;
    std::cout << "DCT Ceptrum: " << t_dctCep << " s , " << t_dctCep*100/totalTime <<"%"<< std::endl;
    std::cout << "Normalization: " << t_norm << " s , " << t_norm*100/totalTime <<"%"<< std::endl;

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::normalization(std::vector<Feature> &normalMels, const std::vector<Feature> & melFes) {
    normalMels.clear();
    if(melFes.size() == 0) return SP_SUCCESS;
    
    Feature means, variance;
    int siz = melFes[0].size();
    means.resize(siz);
    variance.resize(siz);
    for(int i = 0;i < siz;i++) {
        means[i] = variance[i] = 0;
    }

    for(int i = 0;i < melFes.size(); i++) {
        for(int j = 0;j < siz; j++) {
            if(melFes[i].size() > j) {
                means[j] += melFes[i][j];

                variance[j] += melFes[i][j] * melFes[i][j];
            }
        }
    }
    for(int i = 0;i < siz;i++) {
        means[i] /= melFes.size();
        variance[i] /= melFes.size();

        variance[i] = sqrt(variance[i]);
    }

    for(int i = 0;i < melFes.size();i++) {
        normalMels.push_back(melFes[i]);
        for(int j = 0;j < siz;j++) {
            if(j < melFes[i].size()) {
                normalMels[i][j] -= means[j];
                normalMels[i][j] /= variance[j];
            }
        }
    }
        
    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::mel2dct(Feature & feature, std::vector<double> melLog, int cepsNum) {
    int siz = melLog.size();
    feature.resize(siz);
    for(int i = 0;i < siz;i++)
        feature[i] = melLog[i];

//    dct(feature.rawData(), siz, 1);

    dct2(feature.rawData(), siz);

    feature.resize(cepsNum);

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::melCepstrum(std::vector<Feature> &cepstrums, \
        FEATURE_DATA **melLogSpec, \
        int cepsNum) {
    cepstrums.clear();
    
    int framePerBlock = 4;
    
    int rowNum = nfilts, 
        colNum = e_frameNum;
    int elementNum = rowNum * colNum; 
    size_t memSize = elementNum*sizeof(FEATURE_DATA);
    
    FEATURE_DATA * r_melLogSpec_data = (FEATURE_DATA *) malloc(memSize);
    FEATURE_DATA ** r_melLogSpec = (FEATURE_DATA **)malloc(colNum * sizeof(FEATURE_DATA *));
    
    for(int i=0; i<colNum; i++){
        r_melLogSpec[i] = &r_melLogSpec_data[i*rowNum];
    }
    reverseMatrix(r_melLogSpec, melLogSpec, rowNum, colNum);
    
    
    FEATURE_DATA * d_melLogSpec_data;
    
    cudaMalloc((void **) &d_melLogSpec_data, memSize);
    cudaMemcpy(d_melLogSpec_data, r_melLogSpec_data, memSize, cudaMemcpyHostToDevice);

    int blockSize = framePerBlock*rowNum;
    size_t sharedMem = blockSize*sizeof(FEATURE_DATA);
    dim3 dimGrid( ceil((double)elementNum/blockSize) );
    dim3 dimBlock(blockSize);
    mel2dct_kernel<<< dimGrid, dimBlock, sharedMem>>>(d_melLogSpec_data, rowNum, cepsNum);
    cudaMemcpy(r_melLogSpec_data, d_melLogSpec_data, memSize, cudaMemcpyDeviceToHost);

    for(int i=0; i<colNum; i++){
        Feature tmpFeature;
        tmpFeature.resize(cepsNum);
        for(int j=0; j<cepsNum; j++){
           tmpFeature[j] = r_melLogSpec[i][j]; 
        }
        cepstrums.push_back(tmpFeature);
    }
    
    //FEATURE_DATA* e_melCeps_data = (FEATURE_DATA *) malloc(colNum*cepsNum*sizeof(FEATURE_DATA));
    //e_melCeps = (FEATURE_DATA **) malloc(colNum*sizeof(FEATURE_DATA *));
    //size_t copyMemSize = cepsNum*sizeof(FEATURE_DATA);
    //for(int i=0; i<colNum; i++){
    //    e_melCeps[i] = &e_melCeps_data[i*cepsNum];
    //    memcpy(e_melCeps[i], r_melLogSpec[i], copyMemSize);
    //}
    
    cudaFree(d_melLogSpec_data);
    free(r_melLogSpec_data);
    free(r_melLogSpec);

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::reverseMatrix(FEATURE_DATA **outMatrix, FEATURE_DATA **inMatrix, int rowNum, int colNum){
    for(int i=0; i<colNum; i++){
        for(int j=0; j<rowNum; j++){
            outMatrix[i][j] = inMatrix[j][i];
        }
    }
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::powSpectrum(FEATURE_DATA **powSpec, \
        FEATURE_DATA **windows) {
    
    int frameNum = e_frameNum, 
        frameSize = e_frameSize,
        blockSize = e_frameSize,
        elementNum = frameNum * frameSize, 
        selIdx = (int)(std::log2(frameSize))%2;
    
    // Memory Size for whole data
    size_t memSize = elementNum * 2 *sizeof(FEATURE_DATA);
    
    // Share Memory Size in the CUDA
    size_t sharedMem = 2 * blockSize * 2 * sizeof(FEATURE_DATA);

    FEATURE_DATA *SpeechSignal_real = new FEATURE_DATA[elementNum*2], 
                 *d_SpeechSignal_real,
                 *d_SpeechSignal_imag;
    FEATURE_DATA *SpeechSignal_imag = &SpeechSignal_real[elementNum];
    
    // Initialize the Speech Signal by windows (imaginary part are all zero)
    memset(SpeechSignal_real, 0, memSize);
    memcpy(SpeechSignal_real, windows[0], memSize/2);

    
    cudaMalloc( (void **) &d_SpeechSignal_real, memSize );
    cudaMemcpy( d_SpeechSignal_real, SpeechSignal_real, memSize, cudaMemcpyHostToDevice);
    d_SpeechSignal_imag = &d_SpeechSignal_real[elementNum];

    //std::cout << "The select index is: " << selIdx << std::endl;

    dim3 dimGrid( ceil( (double)elementNum/blockSize ) );
    dim3 dimBlock(blockSize);
    windowFFT_kernel<<< dimGrid, dimBlock, sharedMem >>>(d_SpeechSignal_real, d_SpeechSignal_imag, frameNum, frameSize, 1, selIdx);
    cudaMemcpy(SpeechSignal_real, d_SpeechSignal_real, memSize, cudaMemcpyDeviceToHost);
    
    
    // Calculate the Power Spectrum
    int resSize=frameSize/2+1, frameOffset, finalOffset;
    FEATURE_DATA realPart, imagPart;
    e_powFrameSize = resSize;
    
    e_powSpec = (FEATURE_DATA **) malloc(e_frameNum * sizeof(FEATURE_DATA *));
    FEATURE_DATA *tmp_powSpec = (FEATURE_DATA *) malloc(e_frameNum * resSize * sizeof(FEATURE_DATA));
    
    for(int i=0; i<frameNum; i++){
        e_powSpec[i] = &tmp_powSpec[i*resSize];
        frameOffset = i*frameSize;
        for(int j=0; j<resSize; j++){
            finalOffset = frameOffset + j;
            realPart = SpeechSignal_real[finalOffset];
            imagPart = SpeechSignal_imag[finalOffset];
            e_powSpec[i][j] = realPart*realPart + imagPart*imagPart;
        }
    }

    cudaFree(d_SpeechSignal_real);
    delete []SpeechSignal_real;

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::getWts(Matrix<double> &wts, \
        int nfft, \
        double minF, \
        double maxF, \
        int sampleRate, \
        int nfilts, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double)) {

    int nfreqs = nfft / 2 + 1;
    wts.clear();
    std::vector<double> points;

    double minmel = hz2melFunc(minF);
    double maxmel = hz2melFunc(maxF);
    double step = (maxmel - minmel) / (nfilts + 1);
    for(int i = 0; i <= nfilts + 1; i++) 
        points.push_back(mel2hzFunc( minmel + step * i));

    for(int i = 0; i <= nfilts + 1; i++) {
        points[i] = ceil(points[i] / sampleRate * (nfft - 1));
    }
    for(int i = 0;i < nfilts;i++) {
        wts.push_back(std::vector<double>());

        std::vector<double> &filter = wts[i];

        int lp = points[i], mp = points[i+1], rp = points[i+2];
        double lf = 1.0 * points[i] / nfft * sampleRate;
        double mf = 1.0 * points[i+1] / nfft * sampleRate;
        double rf = 1.0 * points[i+2] / nfft * sampleRate;

        while(filter.size() < lp)
            filter.push_back(0.0);

        for(int k = lp;k <= mp;k++) 
            filter.push_back((1.0*k/nfft * sampleRate - lf) / (mf - lf));

        for(int k = mp+1;k <= rp;k++) 
            filter.push_back((rf - 1.0*k/nfft * sampleRate) / (rf - mf));

        while(filter.size() < nfreqs) 
            filter.push_back(0.0);
    }

    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::getWts(FEATURE_DATA ***p_wts, \
        int nfft, \
        double minF, \
        double maxF, \
        int sampleRate, \
        int nfilts, \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double)) {

    int nfreqs = nfft / 2 + 1;
    std::vector<double> points;

    FEATURE_DATA ** wts;
    wts = (FEATURE_DATA **) malloc(nfilts*sizeof(FEATURE_DATA *)); 
    size_t memSize = nfilts * nfreqs * sizeof(FEATURE_DATA);
    FEATURE_DATA * wtsData = (FEATURE_DATA *)malloc(memSize); 
    memset(wtsData,0, memSize);

    double minmel = hz2melFunc(minF);
    double maxmel = hz2melFunc(maxF);
    double step = (maxmel - minmel) / (nfilts + 1);
    for(int i = 0; i <= nfilts + 1; i++) 
        points.push_back(mel2hzFunc( minmel + step * i));

    for(int i = 0; i <= nfilts + 1; i++) {
        points[i] = ceil(points[i] / sampleRate * (nfft - 1));
    }
    for(int i = 0;i < nfilts;i++) {
        wts[i] = &wtsData[i*nfreqs];

        int lp = points[i], mp = points[i+1], rp = points[i+2];
        double lf = 1.0 * points[i] / nfft * sampleRate;
        double mf = 1.0 * points[i+1] / nfft * sampleRate;
        double rf = 1.0 * points[i+2] / nfft * sampleRate;

        for(int k = lp;k <= mp;k++){ 
            wts[i][k] = (1.0*k/nfft * sampleRate - lf) / (mf - lf);
        }
        for(int k = mp+1;k <= rp;k++){ 
            wts[i][k] = (rf - 1.0*k/nfft * sampleRate) / (rf - mf);
        }
        
    }

    e_filterSize = nfreqs;
    e_melWtsExist = true;
    *p_wts = wts;
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::MatrixMul01(FEATURE_DATA ***p_melLog, \
        FEATURE_DATA **wts, \
        FEATURE_DATA **powSpec) {
    FEATURE_DATA *h_melLog, *h_wts, *h_powSpec;
    FEATURE_DATA *d_melLog, *d_wts, *d_powSpec;
    FEATURE_DATA **melLog;
    
    
    size_t memSize1 = nfilts * e_frameNum * sizeof(FEATURE_DATA),
        memSize2 = nfilts * e_filterSize * sizeof(FEATURE_DATA),
        memSize3 = e_frameNum * e_powFrameSize * sizeof(FEATURE_DATA);
    
    h_melLog = (FEATURE_DATA *)malloc(memSize1);
    h_wts = wts[0];
    h_powSpec = powSpec[0];
    
    double startT = wtime();
    
    cudaMalloc((void **)&d_melLog, memSize1);
    cudaMalloc((void **)&d_wts, memSize2);
    cudaMalloc((void **)&d_powSpec, memSize3);
    
    cudaMemcpy(d_wts, h_wts, memSize2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_powSpec, h_powSpec, memSize3, cudaMemcpyHostToDevice);
    
    int bucketNum = (((e_frameNum-1)/BLOCK_SIZE+1)-1)/COL_STEP+1;
    int blockNum = (nfilts-1)/BLOCK_SIZE+1;
    

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(bucketNum,blockNum);
    int r = nfilts, c = e_frameNum;
    
    matrix_mul_kernel<<<dimGrid,dimBlock>>>(d_wts, d_powSpec, d_melLog, r, e_filterSize, c);

    cudaMemcpy(h_melLog, d_melLog, memSize1, cudaMemcpyDeviceToHost);
    
    double endT = wtime();
    //printf("mel filtering calculation time %lf\n", endT-startT);
    
    melLog = (FEATURE_DATA **) malloc(nfilts * sizeof(FEATURE_DATA*));
    for(int i = 0;i < r;i++){
        melLog[i] = &h_melLog[i*c];
    }
    
    *p_melLog = melLog;

    cudaFree(d_melLog);
    cudaFree(d_wts);
    cudaFree(d_powSpec);
    
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::fft2MelLog(int nfft, \
        FEATURE_DATA ***p_melLog,
        FEATURE_DATA **powSpec, \
        int nfilts , \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double), \
        double minF, \
        double maxF, \
        int sampleRate) {
    
    if(!e_melWtsExist){
        getWts(&e_melWts, nfft, minF, maxF, sampleRate, nfilts, hz2melFunc, mel2hzFunc);
    }

    MatrixMul01(p_melLog, e_melWts, powSpec);

    //FEATURE_DATA **melLog = *p_melLog;
    //startT = wtime();
    //for(int i = 0;i < nfilts;i++) 
    //    for(int j = 0;j < e_frameNum;j++){
    //        melLog[i][j] = log(0.0001+fabs(melLog[i][j]));
    //    }
    //finishT = wtime();
    //std::cout << "MelLog: "<<finishT-startT << std::endl;

    return SP_SUCCESS;
}


double FeatureExtractor::preProcessing(FEATURE_DATA **out_windows, \
        const SOUND_DATA *rd, \
        int size, \
        double factor, \
        double winTime, \
        double stepTime, \
        int rate){
    size_empData = size;

    int samplePerWin = ceil(winTime * rate);
    int stepPerWin = ceil(stepTime * rate);
    int nfft = (1 << int(ceil(log(1.0 * samplePerWin)/log(2.0))));
    e_frameSize = nfft;

    e_frameNum = ceil((double)size_empData/stepPerWin);
    size_t winsEleNum = nfft * e_frameNum;
    
    //int paddedSize = nfft*ceil((float)size_empData/stepPerWin)*sizeof(FEATURE_DATA);
    int paddedSize = winsEleNum*sizeof(FEATURE_DATA);
    
    FEATURE_DATA *window_data = (FEATURE_DATA *)malloc(paddedSize);
    memset(window_data, 0, paddedSize);
    
    double startT, finishT, initializeTime;
    startT = wtime();

    FEATURE_DATA *d_window_data;
    cudaMalloc( (void **) &d_window_data, paddedSize );
    cudaMemcpy( d_window_data, window_data, paddedSize, cudaMemcpyHostToDevice );

    SOUND_DATA *d_rd;
    size_t rdMemSize = size*sizeof(SOUND_DATA);
    cudaMalloc( (void **) &d_rd, rdMemSize );
    cudaMemcpy( d_rd, rd, rdMemSize, cudaMemcpyHostToDevice );
    
    finishT = wtime();
    initializeTime = finishT - startT;

    assert(nfft<=1024);
    //std::cout << "nfft: " << nfft << std::endl;
    //size_t sharedMem = nfft*sizeof(FEATURE_DATA);
    dim3 dimGrid( ceil( (double)winsEleNum/nfft) );
    dim3 dimBlock(nfft);
    double arg_PI_factor = 2.0*PI/samplePerWin;
    preProcessing_kernel<<< dimGrid, dimBlock>>>(d_rd, size, d_window_data, samplePerWin, stepPerWin, factor, arg_PI_factor);

    cudaMemcpy(window_data, d_window_data, paddedSize, cudaMemcpyDeviceToHost);

    e_frameNum = ceil((double)size_empData/stepPerWin);
    e_windows = (FEATURE_DATA **)malloc( e_frameNum *sizeof(FEATURE_DATA *));
    for(int i=0,j=0; i<e_frameNum; i++,j+=e_frameSize){
        e_windows[i] = &window_data[j];
    }

    return initializeTime;
}


SP_RESULT FeatureExtractor::windowMul(FEATURE_DATA *window, \
        int size, \
        double (*winFunc)(int, int) ) {
    for(int i = 0;i < size;i++) {
        window[i] *= winFunc(i, size);
    }
    return SP_SUCCESS;
}


SP_RESULT FeatureExtractor::windowing(FEATURE_DATA **out_windows, \
        const FEATURE_DATA *in, \
        double winTime, \
        double stepTime, \
        int rate, \
        double (*winFunc)(int, int)) {
    int samplePerWin = ceil(winTime * rate);
    int stepPerWin = ceil(stepTime * rate);
    int nfft = (1 << int(ceil(log(1.0 * samplePerWin)/log(2.0))));
    e_frameSize = nfft;
    
    int paddedSize = nfft*ceil((float)size_empData/stepPerWin)*sizeof(FEATURE_DATA);
    FEATURE_DATA *window_data = (FEATURE_DATA *)malloc(paddedSize);
    memset(window_data, 0, paddedSize);
    
    int cnt=0, i, j, k;
    for(i = 0, k=0; i < size_empData; i += stepPerWin, k += nfft) {
        cnt++;
        for(j = 0;j < samplePerWin && i+j < size_empData; j++) {
            window_data[k+j] = in[i+j];
        }

        windowMul(&window_data[k],samplePerWin,winFunc);
    }
    
    e_frameNum = cnt;
    e_windows = (FEATURE_DATA **)malloc(cnt*sizeof(FEATURE_DATA *));
    for(i=0,j=0; i<cnt; i++,j+=e_frameSize){
        e_windows[i] = &window_data[j];
    }

    return SP_SUCCESS;
}



SP_RESULT FeatureExtractor::preEmph(/* out */FEATURE_DATA *outs, \
        /*in*/const SOUND_DATA* rd, \
        int size, \
        double factor){
    size_empData = size;
    outs[0]=rd[0];
    for(int i = 1;i<size;i++){
        outs[i]=(1.0 * rd[i] - factor * rd[i-1]);
    }

    return SP_SUCCESS;
}

