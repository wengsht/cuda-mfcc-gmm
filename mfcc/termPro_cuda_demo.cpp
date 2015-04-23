#include "resource.h"
#include "srs.h"
#include <cstdio>
#include <iostream>
#include <fstream>
#include "FeatureExtractor.h"
#include <unistd.h>
#include <cstring>

using namespace std;

int  threadNum = DEFAULT_THREAD_NUM;
bool isCapture;
char wavFileName[1024] = "\0";
void reportMatlab(FeatureExtractor &extractor);
bool dealOpts(int argc, char **argv);

FEATURE_DATA maxEmpData[MAX_BUFFER_SIZE];

int main(int argc, char **argv) {
    if(! dealOpts(argc, argv))
        return 0;

    //cudaDeviceSynchronize();
    
    FeatureExtractor extractor(maxEmpData);

    RawData data;

    // capture mode 
    if(isCapture) 
        capture(wavFileName, data, false);
    else 
        load_wav_file(wavFileName, data);

    extractor.exFeatures(&data);

    reportMatlab(extractor);

    return 0;
}
bool dealOpts(int argc, char **argv) {
    int c;
    while((c = getopt(argc, argv, "c:C:l:L:hj:J")) != -1) {
        switch(c) {
            case 'h':
                printf("usage: \n \
                        filename example: abc\n \
                        Capture Mode: ./pro2_demo -c filename\n \
                        Load    Mode: ./pro2_demo -l filename\n");

                return false;
                break;
            case 'c':
            case 'C':
                isCapture = true;
                strcpy(wavFileName, optarg);
                break;
            case 'l':
            case 'L':
                isCapture = false;
                strcpy(wavFileName, optarg);
                break;
            case 'j':
            case 'J':
                threadNum = atoi(optarg);
            default:
                break;
        }
    }
    if(wavFileName[0] == '\0') return false;

    return true;
}
template <class T> 
void storeVector(const vector<T> &data, const char *filename) {
    ofstream out(filename);
    for(int i = 0;i < data.size(); i++) 
        out << data[i] << endl;
    out.close();
}

void storeBareVector(const FEATURE_DATA *data, int size, const char *filename){
    ofstream out(filename);
    for(int i=0; i<size; i++)
        out << data[i] << endl;
    out.close();
}

void storeBareMatrix(FEATURE_DATA **data, int size1, int size2, const char *filename){
    ofstream out(filename);
    for(int i=0; i<size1; i++){
        for(int j=0; j<size2; j++){
            out << data[i][j] << " ";
        }
        out << endl;
    }
    out.close();
}

template <class T> 
void storeMatrix(const Matrix<T> &data, const char *filename) {
    ofstream out(filename);
    /* 
    if(data.size())
        out << data[0].size() << endl;
        */
    for(int i = 0;i < data.size(); i++) {
        int M = data[i].size();
        for(int j = 0;j < M;j++)
            out << data[i][j] << " ";
        out << endl;
    }
    out.close();
}
void storeFeas(const std::vector<Feature> & data, const char *filename) {
    ofstream out(filename);
    /* 
    if(data.size())
        out << data[0].size() << endl;
        */
    for(int i = 0;i < data.size(); i++) {
        int M = data[i].size();
        for(int j = 0;j < M;j++)
            out << data[i][j] << " ";
        out << endl;
    }
    out.close();
}
void reportMatlab(FeatureExtractor &extractor) {

    //const FEATURE_DATA *externEmpData = extractor.getExEmpData();
    //const int e_SizeEmp = extractor.getSizeEmpData();
    //storeBareVector(externEmpData, e_SizeEmp, "cuda_emp.txt");


    FEATURE_DATA **e_windows = extractor.getExWindows();
    const int tmp_frameNum = extractor.getExFrameNum();
    const int tmp_frameSize = extractor.getExFrameSize();
    //int samplePerWin = ceil(extractor.getWinTime() * extractor.getSampleRate());
    storeBareMatrix(e_windows, tmp_frameNum, tmp_frameSize, "cuda_windows.txt");
    

    FEATURE_DATA **e_powSpec = extractor.getExPowSpec();
    const int tmp_powFrameSize = extractor.getExPowFrameSize();
    storeBareMatrix(e_powSpec, tmp_frameNum, tmp_powFrameSize, "cuda_powSpec.txt");
    

    FEATURE_DATA **e_melLogSpec = extractor.getExMelLogSpec();
    const int tmp_nfilts = extractor.getNfilts();
    storeBareMatrix(e_melLogSpec, tmp_nfilts, tmp_frameNum, "cuda_melLogSpec.txt");


    const vector<Feature> & featrues = extractor.getMelCepstrum();
    storeFeas(featrues, "cuda_melCeps.txt");
   
    //FEATURE_DATA **e_melCeps = extractor.getExMelCeps();
    //const int tmp_cepsNum = extractor.getCepsNum();
    //storeBareMatrix(e_melCeps, tmp_frameNum, tmp_cepsNum, "cuda_melCeps.txt");
    
    const vector<Feature> & normals = extractor.getNormalMelCepstrum();
    storeFeas(normals , "cuda_normalMelCeps.txt");
}
