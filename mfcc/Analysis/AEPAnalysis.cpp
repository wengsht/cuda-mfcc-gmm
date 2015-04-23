//
//  AdaptiveEndPointAnalysis.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "AEPAnalysis.h"




void AEPAnalysis::calcOneBlockBackground(int index) {
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
        double bg = background[index-1];
        double db = energy[index];
        if (db < bg){bg = db;}
        else{bg += (db - bg) * ADJUSTMENT;}
        background.push_back(bg);
    }
}

void AEPAnalysis::calcOneBlockOtherData(int index) {
    if(index < 10){
        level.push_back(energy[index]);
    }
    else{
        double le = level[index-1];
        double bg = background[index-1];
        double db = energy[index];
        le =((le * FORGET_FACTOR) + db) /(FORGET_FACTOR+ 1);
        if (le < bg){le = bg;}
        level.push_back(le);
    }
}

void AEPAnalysis::calcOneBlockSpeech(int index) {
    if(index < 10){
        speech.push_back(false);
    }
    else{
        double le = level[index-1];
        double bg = background[index-1];
        if (le - bg > LEVEL_BG_THRESHOLD){
            speech.push_back(true);
        }
        else{
            speech.push_back(false);
        }
    }
}


int AEPAnalysis::printInf(int from,int end){
    if(end<=from) end  = this->block_num;
    
    for(int i = from;i<end;i++){
        if(speech[i] == true){
            printf(LIGHT_GREEN "%d %lf %lf %lf" NONE"\n",i,energy[i],level[i],background[i]);
        }
        else{
            printf("%d %lf %lf %lf\n",i,energy[i],level[i],background[i]);
        }
    }
    return end;
}

void AEPAnalysis::saveOtherData(FILE * fid){
    saveArray(fid,level.data(),(int)level.size());
}



