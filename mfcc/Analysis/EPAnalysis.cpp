//
//  EndPointAnalysis.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/6/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "EPAnalysis.h"
#include <math.h>
#include <stdio.h>


// no io, using in call back and save
void EPAnalysis::calcOneBlockEnerge(int index){
    assert(this->rawData != NULL);
    
    double sum = 0;
    
    sum = this->rawData->getBlockAveEnergy(index);
    
    energy.push_back(10*log10(sum));
}





// no io, using in call back
bool EPAnalysis::checkContinue(){
    bool inspeech = speech[speech.size()-1];
    if(inspeech){
        silentTime = 1;
    }
    else{
        if(silentTime>=1)
            silentTime++;
    }
    
    if(silentTime >=1 + STOP_IN_SECONDS * FRAME_PER_SECOND){
        return false;
    }
    return true;
}



// no io, using in call back
void EPAnalysis::calcOneBlockData(int index){
    this->calcOneBlockEnerge(index);
    this->calcOneBlockSpeech(index);
    this->calcOneBlockBackground(index);
    this->calcOneBlockOtherData(index);
}

void EPAnalysis::reCalcAllData(){
    Log("====  Caculate energe, background, speech, etc ====");
    this->Initial(this->rawData);
    assert(this->rawData != NULL);
    
    int block_size = this->rawData->getFrameNum() / SAMPLES_IN_EACH_FRAME;
    Log("Now block num = %d",block_size);
    for(int i =0;i<block_size;i++){
        block_num++;
        calcOneBlockData(i);
    }
}

// no io, using in call back
bool EPAnalysis::addOneBlockDataWithEndFlag(const SOUND_DATA * src){
    assert(this->rawData != NULL);
    if(rawData->appendBlockData(src) == false) return false;
    
    calcOneBlockData(this->block_num);
    block_num++;
    return checkContinue();
}


///////////////////////

int EPAnalysis::printInf(int from,int end){
    if(end<=from) end  = this->block_num;
    
    for(int i = from;i<end;i++){
        if(speech[i] == true){
            printf(LIGHT_RED "%d %lf %lf" NONE"\n",i,energy[i],background[i]);
        }
        else{
            printf("%d %lf %lf\n",i,energy[i],background[i]);
        }
    }
    return end;
}

void EPAnalysis::changeSilentSegmentIntoSpeech(){

    ////// move small silent
    Log("Change small silent to speech");
    int p = 0;
    int size = (int)speech.size();

    while(p<size-MIN_SILENT_FRAMES-1){
        if(speech[p] == true && speech[p+1] == false){
            int cnt;
            //for(cnt = 1;cnt<=MIN_SILENT_FRAMES;cnt++){
            for(cnt = 1;cnt<size;cnt++){
                if(speech[p+cnt] == true){
                    break;
                }
            }
            //if(cnt<=MIN_SILENT_FRAMES){
            if(cnt<size){
                for(int i = 1;i<=cnt;i++){
                    speech[p+i] = true;
                }
            }
            p = p+cnt;
        }
        else{
            p++;
        }
    }

}
void EPAnalysis::changeSpeechSegmentIntoSilent(){
    Log("Change small speech to silent");
    
    
    ////// move small talk
    int p = 0;
    int size = (int)speech.size();

    while(p<size-MIN_SPEECH_FRAMES-1){
        if(speech[p] == false && speech[p+1] == true){
            int cnt;
            for(cnt = 1;cnt<=MIN_SPEECH_FRAMES;cnt++){
                if(speech[p+cnt] == false){
                    break;
                }
            }
            if(cnt<=MIN_SPEECH_FRAMES){
                for(int i = 1;i<=cnt;i++){
                    speech[p+i] = false;
                }
            }
            p = p+cnt;
        }
        else{
            p++;
        }
    }

}

void EPAnalysis::smooth(){
    Log("Smooth the label");
    Log("Min silent block %d",MIN_SILENT_FRAMES);
    Log("Min speech block %d",MIN_SPEECH_FRAMES);
    
    changeSilentSegmentIntoSpeech();
//    changeSpeechSegmentIntoSilent();
}

void EPAnalysis::cut(){
    Log("Cut the long silent to short");
    
    int frame_num = this->rawData->getFrameNum();
    int block_num = frame_num/SAMPLES_IN_EACH_FRAME;
    

    int cnt = 0;
    int silent = 0;
    for(int i = 0;i<block_num;i++){
        if(speech[i]){
            rawData->copyBlockData(i, cnt);
            cnt++;
            silent = 0;
        }
        else{
            silent++;
            if(silent>MIN_SILENT_FRAMES*2){
                continue;
            }
            else{
                rawData->copyBlockData(i, cnt);
                cnt++;
            }
        }
    }
    Log("From %d block to %d block",block_num,cnt);
    this->rawData->setFrameNum(cnt*SAMPLES_IN_EACH_FRAME);
    
    this->reCalcAllData();
    
}


bool EPAnalysis::saveMatlab(const char * file_name){
    Tip("Save data to %s",file_name);
    FILE  *fid;
    
    fid = fopen(file_name, "w");
    if( fid == NULL )
    {
        ErrorLog("Could not open file: %s\n",file_name);
        return false;
    }
    else
    {

        saveArray(fid,energy.data(),(int)energy.size());

        saveArray(fid,speech.data(),(int)speech.size());

        saveArray(fid,background.data(),(int)background.size());

        //fprintf(fid,"y2 = (y2 + 1)*50\n");
        //fprintf(fid,"figure; hold on;\n");
        //fprintf(fid,"plot(x,db)\n");
        //fprintf(fid,"plot(x,sp,'m')\n");
        //fprintf(fid,"plot(x,bg,'g')\n");

        saveOtherData(fid);
        
        fclose( fid );
    }
    return true;
}











