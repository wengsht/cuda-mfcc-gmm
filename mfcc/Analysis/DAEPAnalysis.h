//
//  DAEPAnalysis.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__DAEPAnalysis__
#define __SpeechRecongnitionSystem__DAEPAnalysis__

#include <iostream>
#include "AEPAnalysis.h"


class DAEPAnalysis:public EPAnalysis{
protected:
    virtual void calcOneBlockBackground(int index) ;
    virtual void calcOneBlockSpeech(int index) ;
public:
    //virtual int printInf(int from,int end = -1);

    DAEPAnalysis(){};
    ~DAEPAnalysis(){};
    
};

#endif /* defined(__SpeechRecongnitionSystem__DAEPAnalysis__) */
