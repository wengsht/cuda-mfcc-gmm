//
//  FeatureExtractor.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__FeatureExtractor__
#define __SpeechRecongnitionSystem__FeatureExtractor__

#include <iostream>
#include <vector>
#include <cmath>
#include "mathtool.h"
#include "Feature.h"
#include "RawData.h"
#include "resource.h"
#include <unistd.h>
#include <pthread.h>

class FeatureExtractor{
    CONST_REFERENCE_READ_ONLY_DECLARE(std::vector<double> , emp_data, EmpData)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, size_empData, SizeEmpData)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, e_frameNum, ExFrameNum)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, e_frameSize, ExFrameSize)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, e_powFrameSize, ExPowFrameSize)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, e_melSize, ExMelSize)
    CONST_REFERENCE_READ_ONLY_DECLARE(int, e_filterSize, ExFilterSize)
    CONST_REFERENCE_READ_ONLY_DECLARE(bool, e_melWtsExist, ExMelWtsExist)
    CONST_REFERENCE_READ_ONLY_DECLARE(bool, isCuda, CudaFlag)
    CONST_REFERENCE_READ_ONLY_DECLARE(Matrix<double> , windows, Windows)
    CONST_REFERENCE_READ_ONLY_DECLARE(Matrix<double> , powSpec, PowSpectrum)
    CONST_REFERENCE_READ_ONLY_DECLARE(Matrix<double> , melLogSpec, MelLogSpec)

    CONST_REFERENCE_READ_ONLY_DECLARE(std::vector<Feature>, melCeps, MelCepstrum);
    CONST_REFERENCE_READ_ONLY_DECLARE(std::vector<Feature>, normalMelCeps, NormalMelCepstrum);

    READ_WRITE_DECLARE(FEATURE_DATA *, e_emp_data, ExEmpData)
    READ_WRITE_DECLARE(FEATURE_DATA **, e_windows, ExWindows)
    READ_WRITE_DECLARE(FEATURE_DATA **, e_powSpec, ExPowSpec)
    READ_WRITE_DECLARE(FEATURE_DATA **, e_melLogSpec, ExMelLogSpec)
    READ_WRITE_DECLARE(FEATURE_DATA **, e_melWts, ExMelWts)
    READ_WRITE_DECLARE(FEATURE_DATA **, e_melCeps, ExMelCeps)
    READ_WRITE_DECLARE(FEATURE_DATA *, e_delta1, ExDelta1)
    READ_WRITE_DECLARE(FEATURE_DATA *, e_delta2, ExDelta2)
    READ_WRITE_DECLARE(int , sampleRate, SampleRate);
    READ_WRITE_DECLARE(double , preEmpFactor, PreEmpFactor);
    READ_WRITE_DECLARE(double, winTime, WinTime);
    READ_WRITE_DECLARE(double, stepTime, StepTime);
    READ_WRITE_DECLARE(double, minF, MinF);
    READ_WRITE_DECLARE(double, maxF, MaxF);
    READ_WRITE_DECLARE(int, nfilts, Nfilts);
    READ_WRITE_DECLARE(int, cepsNum, CepsNum);

private:
    double (*winFunc)(int, int);
    double (*hz2melFunc)(double);
    double (*mel2hzFunc)(double);

protected:

//    std::vector<Feature> melCeps;
    void inital(){
        melCeps.clear();
        emp_data.clear();
        melLogSpec.clear();
        powSpec.clear();
        windows.clear();
    }
    
    // pre emphasize and save the data into emp_data
    // if factor == 0 then no emphasize
    SP_RESULT preEmph(std::vector<double> &outs, \
            const SOUND_DATA* rd, \
            int size , \
            double factor = SP_PREEMPH_FACTOR);

    SP_RESULT preEmph(double* outs, \
            const SOUND_DATA* rd, \
            int size , \
            double factor = SP_PREEMPH_FACTOR);
    
    SP_RESULT windowing(Matrix<double> & out_windows, \
            const std::vector<double> &in, \
            double winTime = WINTIME, \
            double stepTime = STEPTIME, \
            int rate = SAMPLE_RATE, \
            double (*winFunc)(int, int) = FeatureExtractor::hanning);
    

    SP_RESULT windowing(FEATURE_DATA** out_windows, \
            const FEATURE_DATA *in, \
            double winTime = WINTIME, \
            double stepTime = STEPTIME, \
            int rate = SAMPLE_RATE, \
            double (*winFunc)(int, int) = FeatureExtractor::hanning);

    double preProcessing(FEATURE_DATA** out_windows, \
            const SOUND_DATA* rd, \
            int size, \
            double factor = SP_PREEMPH_FACTOR, \
            double winTime = WINTIME, \
            double stepTime = STEPTIME, \
            int rate = SAMPLE_RATE);

    SP_RESULT fftPadding(Matrix<double> & out_pads);
    
    SP_RESULT powSpectrum(Matrix<double> &powSpectrum, Matrix<double> &windows);
    //SP_RESULT powSpectrum(FEATURE_DATA **powSpectrum, FEATURE_DATA **windows);
    SP_RESULT powSpectrum(FEATURE_DATA **powSpectrum, FEATURE_DATA **windows);

    SP_RESULT melCepstrum(std::vector<Feature> &cepstrums, \
            const Matrix<double> &melLogSpec, \
            int cepsNum = CEPS_NUM);
    
    SP_RESULT melCepstrum(std::vector<Feature> &cepstrums, \
            FEATURE_DATA **melLogSpec, \
            int cepsNum = CEPS_NUM);

    SP_RESULT reverseMatrix(FEATURE_DATA **outMatrix, FEATURE_DATA **inMatrix, int rowNum, int colNum);

    void windowFFT(std::vector<double> &res, \
            std::vector<double> &data);


    static double hanning(int n, int M) {
        return 0.5 - 0.5 * cos (2.0 * PI * n / M);
    }
    static double hz2mel(double frequency) {
        return 2595.0 * log(1+frequency/700.0) / log(10.0);
    }
    static double mel2hz(double hz) {
        return 700.0* ( pow(10.0, hz/2595.0) - 1.0);
    }
    static double getDB(double pow) {
        return 10.0 * log(pow) / log(10.0);
    }
    /* 
     * melLog = wts * powSpec'
     * */
    SP_RESULT MatrixMul01(Matrix<double> & melLog, \
            Matrix<double> &wts, \
            Matrix<double> & powSpec);

    SP_RESULT MatrixMul01(FEATURE_DATA ***p_melLog, \
            FEATURE_DATA **wts, \
            FEATURE_DATA **powSpec);

    SP_RESULT fft2MelLog(int nfft, \
            Matrix<double> &melLog, \
            Matrix<double> & powSpec, \
            int nfilts = MEL_FILTER_NUM, \
            double (*hz2melFunc)(double) = FeatureExtractor::hz2mel, \
            double (*mel2hzFunc)(double) = FeatureExtractor::mel2hz, \
            double minF = MIN_F, \
            double maxF = MAX_F, \
            int sampleRate = SAMPLE_RATE);

    
    SP_RESULT fft2MelLog(int nfft, \
            FEATURE_DATA ***p_melLog, \
            FEATURE_DATA **powSpec, \
            int nfilts = MEL_FILTER_NUM, \
            double (*hz2melFunc)(double) = FeatureExtractor::hz2mel, \
            double (*mel2hzFunc)(double) = FeatureExtractor::mel2hz, \
            double minF = MIN_F, \
            double maxF = MAX_F, \
            int sampleRate = SAMPLE_RATE);


    SP_RESULT mel2dct(Feature & feature, std::vector<double> melLog, int cepsNum = CEPS_NUM);

    SP_RESULT normalization(std::vector<Feature> &normalMels,const std::vector<Feature> & melFes);

//    SP_RESULT getMelLog(std::vector<double> & melLog, \
            const std::vector<double> & powSpec, \
            const Matrix<double> &wts);

    SP_RESULT getWts(Matrix<double> &wts, \
            int nfft, \
            double minF = MIN_F, \
            double maxF = MAX_F, \
            int sampleRate = SAMPLE_RATE, \
            int nfilts = MEL_FILTER_NUM, \
            double (*hz2melFunc)(double) = FeatureExtractor::hz2mel, \
            double (*mel2hzFunc)(double) = FeatureExtractor::mel2hz);

    SP_RESULT getWts(FEATURE_DATA ***p_wts, \
            int nfft, \
            double minF = MIN_F, \
            double maxF = MAX_F, \
            int sampleRate = SAMPLE_RATE, \
            int nfilts = MEL_FILTER_NUM, \
            double (*hz2melFunc)(double) = FeatureExtractor::hz2mel, \
            double (*mel2hzFunc)(double) = FeatureExtractor::mel2hz);
    

    SP_RESULT windowMul(std::vector<double> &window, \
            double (*winFunc)(int, int) );
    
    SP_RESULT windowMul(FEATURE_DATA *window, \
            int size, \
            double (*winFunc)(int, int) );
    
public:
    FeatureExtractor() : \
            isCuda(false), \
            threadNum(DEFAULT_THREAD_NUM), \
            sampleRate(SAMPLE_RATE), \
            preEmpFactor(SP_PREEMPH_FACTOR), \
            winTime(WINTIME), \
            stepTime(STEPTIME), \
            winFunc(FeatureExtractor::hanning), \
            minF(MIN_F), \
            maxF(MAX_F), \
            hz2melFunc(FeatureExtractor::hz2mel), \
            mel2hzFunc(FeatureExtractor::mel2hz), \
            nfilts(MEL_FILTER_NUM), \
            cepsNum(CEPS_NUM) {}
    FeatureExtractor(FEATURE_DATA *maxEmpData) : \
            isCuda(true), \
            e_emp_data(maxEmpData), \
            e_melWtsExist(MEL_WTS_EXIST), \
            threadNum(DEFAULT_THREAD_NUM), \
            sampleRate(SAMPLE_RATE), \
            preEmpFactor(SP_PREEMPH_FACTOR), \
            winTime(WINTIME), \
            stepTime(STEPTIME), \
            winFunc(FeatureExtractor::hanning), \
            minF(MIN_F), \
            maxF(MAX_F), \
            hz2melFunc(FeatureExtractor::hz2mel), \
            mel2hzFunc(FeatureExtractor::mel2hz), \
            nfilts(MEL_FILTER_NUM), \
            cepsNum(CEPS_NUM) {}
    FeatureExtractor(int threadNum) : \
            isCuda(false), \
            threadNum(threadNum), \
            sampleRate(SAMPLE_RATE), \
            preEmpFactor(SP_PREEMPH_FACTOR), \
            winTime(WINTIME), \
            stepTime(STEPTIME), \
            winFunc(FeatureExtractor::hanning), \
            minF(MIN_F), \
            maxF(MAX_F), \
            hz2melFunc(FeatureExtractor::hz2mel), \
            mel2hzFunc(FeatureExtractor::mel2hz), \
            nfilts(MEL_FILTER_NUM), \
            cepsNum(CEPS_NUM) {}
    ~FeatureExtractor() {
        if(isCuda){
            free(e_windows[0]);
            free(e_windows);
            free(e_powSpec[0]);
            free(e_powSpec);
            free(e_melWts[0]);
            free(e_melWts);
            free(e_melLogSpec[0]);
            free(e_melLogSpec);
            //free(e_melCeps[0]);
            //free(e_melCeps);
            //free(e_delta1);
            //free(e_delta2);
        }
    }

    void doubleDelta(std::vector<Feature> &normalMelCeps);
    
//    void calcFeatures(const RawData* rd);
    
    SP_RESULT exFeatures(const RawData *data, \
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
            int cepsNum);

    // normalMelCeps will be 13 length
    SP_RESULT exFeatures(const RawData *data);

    // normalMelCeps will be 39 length vector now
    //
    SP_RESULT exDoubleDeltaFeatures(const RawData *data);

private:
    int threadNum;

};
#endif /* defined(__SpeechRecongnitionSystem__FeatureExtractor__) */
