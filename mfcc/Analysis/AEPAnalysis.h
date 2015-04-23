//
//  AdaptiveEndPointAnalysis.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__AdaptiveEndPointAnalysis__
#define __SpeechRecongnitionSystem__AdaptiveEndPointAnalysis__

#include <iostream>
#include "EPAnalysis.h"



class AEPAnalysis:public EPAnalysis{
protected:
    vector<double> level;
    virtual void saveOtherData(FILE * fid);

public:
    AEPAnalysis(){
        level.clear();
    }
    ~AEPAnalysis(){};
    
    virtual int printInf(int from,int end = -1);

    
    virtual void Initial(RawData * rawData){
        EPAnalysis::Initial(rawData);
        level.clear();
    }
    
    virtual void calcOneBlockBackground(int index) ;
    virtual void calcOneBlockSpeech(int index) ;
    virtual void calcOneBlockOtherData(int index) ;


};
#endif /* defined(__SpeechRecongnitionSystem__AdaptiveEndPointAnalysis__) */
