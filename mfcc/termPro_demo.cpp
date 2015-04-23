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

//FEATURE_DATA maxEmpData[MAX_BUFFER_SIZE];

int main(int argc, char **argv) {
    if(! dealOpts(argc, argv))
        return 0;

    //cudaDeviceSynchronize();
    
    FeatureExtractor extractor(threadNum);

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
    const vector<double> &empData = extractor.getEmpData();
    storeVector(empData, "emp.txt");

    const Matrix<double> &windows = extractor.getWindows();
    storeMatrix(windows, "windows.txt");

    const Matrix<double> &powSpec = extractor.getPowSpectrum();
    storeMatrix(powSpec, "powSpec.txt");

    const Matrix<double> &melLog = extractor.getMelLogSpec();
    storeMatrix(melLog, "melLogSpec.txt");
    
    const vector<Feature> & featrues = extractor.getMelCepstrum();
    storeFeas(featrues, "melCeps.txt");

    const vector<Feature> & normals = extractor.getNormalMelCepstrum();
    storeFeas(normals , "normalMelCeps.txt");
}
