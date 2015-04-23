//
//  BEPAnalysis.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/8/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__BEPAnalysis__
#define __SpeechRecongnitionSystem__BEPAnalysis__

#include "EPAnalysis.h"


#include <vector>
using namespace std;

//Energy-based endpointing
class BEPAnalysis:public EPAnalysis{
    
protected:

    virtual void calcOneBlockBackground(int index) ;
    virtual void calcOneBlockSpeech(int index) ;
    

public:
    BEPAnalysis(){}
    ~BEPAnalysis(){};
};

#endif /* defined(__SpeechRecongnitionSystem__BEPAnalysis__) */
