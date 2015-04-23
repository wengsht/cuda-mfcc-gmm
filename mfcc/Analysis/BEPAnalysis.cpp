//
//  BEPAnalysis.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/8/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "BEPAnalysis.h"


void BEPAnalysis::calcOneBlockBackground(int index){
    background.push_back(SILENCE_ENERGY_THRESHOLD);
}

void BEPAnalysis::calcOneBlockSpeech(int index){
    if(energy[index] > SILENCE_ENERGY_THRESHOLD) {
        speech.push_back(true);
    }
    else{
        speech.push_back(false);
    }
}

