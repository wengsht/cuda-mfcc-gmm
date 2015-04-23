//
//  EndPointAnalysis.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/6/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__EndPointAnalysis__
#define __SpeechRecongnitionSystem__EndPointAnalysis__


#include "RawData.h"
#include <vector>
using namespace std;

//Energy-based endpointing
class EPAnalysis{
    
protected:
    int block_num ;
    int silentTime; // don't find how many block
    
    READ_WRITE_DECLARE(RawData*,rawData,RawData);
    vector<double> background;
    // the decibel of each Block
    // there are energy.size() = FrameIndex / FRAMES_IN_EACH_BUFFER_FOR_CAPTURE blocks
    vector<double> energy;
    vector<char> speech;
    
    // calc and pushback
    virtual void calcOneBlockEnerge(int index);
    virtual void calcOneBlockBackground(int index) =0;
    virtual void calcOneBlockSpeech(int index) =0;
    virtual void calcOneBlockOtherData(int index) {};
    
    void calcOneBlockData(int index);

    bool appendOneBlockData(const SOUND_DATA * src);
    
    virtual bool checkContinue();
    virtual void saveOtherData(FILE * fid){};
    
    void changeSilentSegmentIntoSpeech();
    void changeSpeechSegmentIntoSilent();
public:
    EPAnalysis(){
        Initial(NULL);
    }
    
    ~EPAnalysis(){};
    
    RawData * data() {
        return this->rawData;
    }
    virtual void Initial(RawData * rawData){
        this->rawData = rawData;
        energy.clear();
        speech.clear();
        background.clear();
        silentTime = 0;
        block_num = 0;
    }
    

    
    bool addOneBlockDataWithEndFlag(const SOUND_DATA * src);
    
    void reCalcAllData();
    
    void smooth();
    void cut();
    // return the end of the printed position as the next start position
    // end = -1 mean print all information
    // do not use in callback
    virtual int printInf(int from,int end = -1);
    
    //save data to show in matlab 
    virtual bool saveMatlab(const char * file_name);
    
};
#endif /* defined(__SpeechRecongnitionSystem__EndPointAnalysis__) */
