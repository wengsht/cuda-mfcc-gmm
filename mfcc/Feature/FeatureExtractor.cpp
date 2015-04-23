#include "FeatureExtractor.h"
#include "RawData.h"
#include <cmath>
#include <ctime>
#include <cstdlib>
#include "wtime.h"
#include "mathtool.h"
#include "ThreadPool.h"

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
    SP_RESULT res; 
    inital();
    
    double startT, finishT;
    double totalTime = 0;

    startT = wtime();
    res = preEmph(emp_data, data->getData(), data->getFrameNum(), preEmpFactor);
    finishT = wtime();
    double t_preemp = finishT-startT;
    totalTime += t_preemp;

    startT = wtime();
    res = windowing(windows, emp_data, winTime, stepTime, sampleRate, winFunc);
    finishT = wtime();
    double t_window = finishT-startT;
    totalTime += t_window;

    startT = wtime();
    fftPadding(windows);
    finishT = wtime();
    double t_fftpad = finishT-startT;
    totalTime += t_fftpad;

    startT = wtime();
    powSpectrum(powSpec, windows);
    finishT = wtime();
    double t_powSpec = finishT-startT;
    totalTime += t_powSpec;

    if(powSpec.size() == 0) return SP_SUCCESS;

    int nfft = (powSpec[0].size() -1) << 1;

    startT = wtime();
    fft2MelLog(nfft, melLogSpec, powSpec, nfilts, hz2melFunc, mel2hzFunc, minF, maxF, sampleRate);
    finishT = wtime();
    double t_mel = finishT-startT;
    totalTime += t_mel;

    startT = wtime();
    melCepstrum(melCeps, melLogSpec, cepsNum);
    finishT = wtime();
    double t_dctCep = finishT-startT;
    totalTime += t_dctCep;

    startT = wtime();
    normalization(normalMelCeps, melCeps);
    finishT = wtime();
    double t_norm = finishT-startT;
    totalTime += t_norm;

    doubleDelta(normalMelCeps);

    std::cout << "Total Time: " << totalTime << std::endl;
    std::cout << "PreEmp: " << t_preemp << " s , " << t_preemp*100/totalTime <<"%" <<std::endl;
    std::cout << "Windowing: " << t_window << " s , " << t_window*100/totalTime <<"%" << std::endl;
    std::cout << "FFT padding: " << t_fftpad << " s , " << t_fftpad*100/totalTime <<"%"<< std::endl;
    std::cout << "PowerSpectrum: " << t_powSpec << " s , " << t_powSpec*100/totalTime <<"%"<< std::endl;
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

    //for(int i=0; i<siz; i++) std::cout << feature[i] << " ";
    //std::cout << std::endl;
    
//    dct(feature.rawData(), siz, 1);

    dct2(feature.rawData(), siz);

    //for(int i=0; i<siz; i++) std::cout << feature[i] << " ";
    //std::cout << std::endl;
    
    feature.resize(cepsNum);

    //for(int i=0; i<cepsNum; i++) std::cout << feature[i] << " ";
    //std::cout << std::endl << std::endl;

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::melCepstrum(std::vector<Feature> &cepstrums, \
        const Matrix<double> &melLogSpec, \
        int cepsNum) {
    cepstrums.clear();
    if(melLogSpec.size() == 0) return SP_SUCCESS;

    for(int i = 0;i < melLogSpec[0].size(); i++) {
        std::vector<double> tmp;
        for(int j = 0;j < melLogSpec.size(); j++)
            if(melLogSpec[j].size() > i)
                tmp.push_back(melLogSpec[j][i]);

        cepstrums.push_back(Feature());

        mel2dct(cepstrums[i], tmp, cepsNum);
    }
    return SP_SUCCESS;
}

/*
void FeatureExtractor::fftTask(void *in) {
    fft_task_info * task_info = (fft_task_info *) in;

    std::vector<double> &window = *(task_info->window);
    std::vector<double> &powSpec = *(task_info->powWinSpec);

    windowFFT(powSpec, window);

    delete task_info;
}
*/

SP_RESULT FeatureExtractor::powSpectrum(Matrix<double> &powSpec, \
        Matrix<double> &windows) {
    if(windows.size() == 0) return SP_SUCCESS;

    powSpec.resize(windows.size());
    int siz = windows[0].size();
    std::vector<double> powWinSpec(windows[0].size());

    
    for(int i = 0;i < windows.size(); i++) {
        if(windows[i].size() != siz) continue;
        //powSpec.push_back(windowFFT(powWinSpec, windows[i]));
        windowFFT(powSpec[i], windows[i]);
    }
    
    /*
    ThreadPool threadPool(threadNum);
    for(int i = 0;i < windows.size();i++) {
        sp_task task;

        if(windows[i].size() != siz) continue;

        fft_task_info *task_info = new fft_task_info;
        task_info->window = &(windows[i]);
        task_info->powWinSpec = &(powSpec[i]);

        task.func = fftTask;
        task.in   = task_info;

        threadPool.addTask(task);
    }
    threadPool.run();
    */

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

/*  
SP_RESULT FeatureExtractor::getMelLog(std::vector<double> & melLog, \
        const std::vector<double> & powSpec, \
        const Matrix<double> &wts) {
    melLog.resize(powSpec.size());
    for(int i = 0;i < melLog.size(); i++) melLog[i] = 0.0;
    for(int i = 0;i < wts.size();i++) {
        int mxSiz = std::min(wts[i].size(), powSpec.size());

        for(int j = 0;j < mxSiz;j++) 
            melLog[j] += powSpec[j] * wts[i][j];
    }
    for(int i = 0;i < melLog.size(); i++) 
        melLog[i] = getDB(melLog[i]);

    return SP_SUCCESS;
}
*/

/*
void FeatureExtractor::mulTask(void *in) {
    mul_task_info * task_info = (mul_task_info *) in;

    std::vector<double> &melLog = *(task_info->melLog);
    std::vector<double> &wts = *(task_info->wts);
    Matrix<double> &powSpec = *(task_info->powSpec);

    for(int j = 0;j < powSpec.size();j++) {
        melLog[j] = 0.0;
        int mx = std::min(wts.size(), powSpec[j].size());
        for(int k = 0;k < mx;k++)
            melLog[j] += wts[k] * powSpec[j][k];
    }

    delete task_info;
}
*/

SP_RESULT FeatureExtractor::MatrixMul01(Matrix<double> & melLog, \
        Matrix<double> &wts, \
        Matrix<double> & powSpec) {

    int r = wts.size(), c = powSpec.size();

    melLog.resize(r);
    for(int i = 0;i < r;i++)
        melLog[i].resize(c);

    
    for(int i = 0;i < r;i++) {
        for(int j = 0;j < c;j++) {
            melLog[i][j] = 0.0;
            int mx = std::min(wts[i].size(), powSpec[j].size());
            for(int k = 0;k < mx;k++)
                melLog[i][j] += wts[i][k] * powSpec[j][k];
        }
    }
    
    /*  
    ThreadPool threadPool(threadNum);

    for(int i = 0;i < r;i++) {
        struct sp_task task_struct;
        struct mul_task_info *task_info = new mul_task_info;
        
        task_info->wts = &(wts[i]);
        task_info->powSpec = &powSpec;
        task_info->melLog = &(melLog[i]);

        task_struct.func = mulTask;
        task_struct.in   = task_info;

        threadPool.addTask(task_struct);
    }
    threadPool.run();
    */
    
    return SP_SUCCESS;
}
SP_RESULT FeatureExtractor::fft2MelLog(int nfft, \
        Matrix<double> &melLog, \
        Matrix<double> & powSpec, \
        int nfilts , \
        double (*hz2melFunc)(double), \
        double (*mel2hzFunc)(double), \
        double minF, \
        double maxF, \
        int sampleRate) {
    Matrix<double> wts;
    getWts(wts, nfft, minF, maxF, sampleRate, nfilts, hz2melFunc, mel2hzFunc);

    melLog.clear();

    MatrixMul01(melLog, wts, powSpec);

    for(int i = 0;i < melLog.size();i++) 
        for(int j = 0;j < melLog[i].size();j++)
            melLog[i][j] = log(0.0001+fabs(melLog[i][j]));

    return SP_SUCCESS;
}

void FeatureExtractor::windowFFT(std::vector<double> &res, \
        std::vector<double> &data) {
    res.resize(data.size() / 2 + 1);
    std::complex<double> * cp = new std::complex<double>[data.size()];

    for(int i = 0;i < data.size();i++) {
        cp[i] = std::complex<double>(data[i], 0);
    }

    fft(cp, data.size(), 1);
    //fft(cp, data.size(), -1);
    //dft(cp, data.size(), 1);

    for(int i = 0;i < res.size();i++) {
        res[i] = std::norm(cp[i]);
    }

    delete [] cp;

}

SP_RESULT FeatureExtractor::windowMul(std::vector<double> &window, \
        double (*winFunc)(int, int) ) {
    int M = window.size();
    for(int i = 0;i < M;i++) {
        window[i] *= winFunc(i, M);
    }
    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::windowing(Matrix<double> & out_windows, \
        const std::vector<double> & in, \
        double winTime, \
        double stepTime, \
        int rate, \
        double (*winFunc)(int, int)) {
    int samplePerWin = ceil(winTime * rate);
    int stepPerWin = ceil(stepTime * rate);
    
//    int nfft = 2 ^ (ceil(log(1.0 * samplePerWin)/log(2.0)));
    //std::cout << "Total Size: " << in.size() << std::endl;
    //std::cout << "SamplePerWin : " << samplePerWin << std::endl;
    //std::cout << "stepPerWin: " << stepPerWin << std::endl;
    std::vector<double> buf(samplePerWin);
    int i,j;
    for(i = 0; i < in.size(); i += stepPerWin) {
        for(j = 0;j < samplePerWin && i+j < in.size(); j++) {
            buf[j] = in[i+j];
        }
        //std::cout << "Inner Size: " << j << std::endl;
        if(j<buf.size()){
            for(int k=j; k<buf.size(); k++){
                buf[k]=0;
            }
        }
        windowMul(buf, winFunc);

        out_windows.push_back(buf);
    }

    return SP_SUCCESS;
}

SP_RESULT FeatureExtractor::preEmph(/* out */std::vector<double> &outs, \
        /*in*/const SOUND_DATA* rd, \
        int size, \
        double factor){
    outs.clear();
    outs.push_back(rd[0]);
    for(int i = 1;i<size;i++){
        outs.push_back(1.0 * rd[i] - factor * rd[i-1]);
    }

    return SP_SUCCESS;
}

/*
void FeatureExtractor::paddingTask(void *in) {
    padding_task_info * info = (padding_task_info *) in;

    std::vector<double> & window = *(info->window);
    int nfft = info->nfft;

    while(window.size() < nfft) { 
        window.push_back(0.0);
    }

    delete info;
}
*/
SP_RESULT FeatureExtractor::fftPadding(Matrix<double> & windows) {
    if(windows.size() == 0) return SP_SUCCESS;
    int samplePerWin = windows[0].size();

    int nfft = (1 << int(ceil(log(1.0 * samplePerWin)/log(2.0))));

    
    for(int i = 0;i < windows.size();i++) {
        while(windows[i].size() < nfft) 
            windows[i].push_back(0.0);
    }
    
    /* 
    ThreadPool threadPool(threadNum);
    for(int i = 0;i < windows.size();i++) {
        struct sp_task task_struct;
        struct padding_task_info *task_info = new padding_task_info;
        
        task_info->window = &(windows[i]);
        task_info->nfft = nfft;

        task_struct.func = paddingTask;
        task_struct.in   = task_info;

        threadPool.addTask(task_struct);
    }
    threadPool.run();
    */

    return SP_SUCCESS;
}
