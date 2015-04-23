//
//  DAEPAnalysis.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "DAEPAnalysis.h"
void DAEPAnalysis::calcOneBlockBackground(int index){
    if(index < 10){
        if(index == 0)
            background.push_back(energy[0]);
        else{
            double ave_energe = background[index-1];
            ave_energe = (ave_energe * (index) + energy[index])/(index+1);
            background.push_back(ave_energe);
        }
    }
    else{
        double db = energy[index];
        double inspeech = speech[index];
        double bg = background[index-1];
        
        if (inspeech){
            bg += (db-bg)*TRACKING_FACTOR;
        }
        else{
            bg= ALPHA*bg + (1-ALPHA)*db;
        }
        background.push_back(bg);
    }
}

void DAEPAnalysis::calcOneBlockSpeech(int index){
    if(index < 10){
        speech.push_back(false);
    }
    else{
        double db = energy[index];
        double bg = background[index-1];
        double inspeech = speech[index-1];
        if (inspeech){
            if (db-bg < OFFSET_THRESHOLD){inspeech = 0;}
        }
        else {
            if (db-bg > ONSET_THRESHOLD){inspeech = 1;}
        }
        speech.push_back(inspeech);
    }
}


