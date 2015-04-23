//
//  TimeCapture.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/5/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef SpeechRecongnitionSystem_TimeCapture_h
#define SpeechRecongnitionSystem_TimeCapture_h
#include "Capture.h"
#include "EPAnalysis.h"

// define and use a call back function
class AutoCapture:public Capture{
protected:
    EPAnalysis * ep;
public:
    AutoCapture(EPAnalysis * ep):ep(ep){};
    ~AutoCapture(){};
    bool captureAction(RawData * data);
    bool init_callback(RawData * data,bool input);
};


#endif
