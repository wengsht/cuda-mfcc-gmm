//
//  RawData.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/26/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "RawData.h"

RawData::RawData()
{
    Log("Create RawData");
    this->data = NULL;
    this->data = new SOUND_DATA[MAX_SIZE];
    if(this->data == NULL){
        ErrorLog("No enough room");
    }
    else{
        Log("Get room successfully");
    }
    clean();
}

RawData::RawData(const RawData& rawData){
    this->frame_num = rawData.frame_num;
    this->data = NULL;
    this->data = new SOUND_DATA[MAX_SIZE];
    if(this->data == NULL){
        ErrorLog("No enough room");
    }
    else{
        Log("Get room successfully");
    }
}

RawData::~RawData(){
    Log("Delete RawData");
    SAFE_DELETE_POINTER(this->data);
    Log("Delete Safely");
}


void RawData::setData(int index,SOUND_DATA d){
    this->data[index] = d;
}


void RawData::clean(){
    Log("Clean RawData");
    for(int i = 0;i<MAX_SIZE;i++){
        this->data[i] = 0;
    }
    setFrameNum(0);
}

const SOUND_DATA * RawData::getData()const{
    return this->data;
}

bool RawData::appendBlockData(const SOUND_DATA * src){
    if(frame_num+SAMPLES_IN_EACH_FRAME>=MAX_SIZE)return false;
    
    for(int i = 0;i<SAMPLES_IN_EACH_FRAME;i++){
        this->data[frame_num+i] = src[i];
    }
    frame_num += SAMPLES_IN_EACH_FRAME;
    return true;
}

void RawData::copyBlockData(int from,int to){
    from = from * SAMPLES_IN_EACH_FRAME;
    to = to *SAMPLES_IN_EACH_FRAME;
    if(from >= frame_num)return;
    if(to>=frame_num)return;
	if(from == to) return;
    for(int i = 0;i<SAMPLES_IN_EACH_FRAME;i++){
        data[to+i] = data[from+i];
    }
}


double RawData::getBlockAveEnergy(int index){
    double ret = 0;
    int st = index *SAMPLES_IN_EACH_FRAME;
    
    // overflow
    if(st > frame_num){
        WarnLog("Calculate average energy in %d block, max %d",
                index,frame_num/SAMPLES_IN_EACH_FRAME);
        return ret;
    }
    
    //calculating
    for(int i = 0;i<SAMPLES_IN_EACH_FRAME;i++){
        double x = data[st+i];
        ret +=  x*x;
    }
    ret /= SAMPLES_IN_EACH_FRAME;
    return ret;
}

bool RawData::setFrameNum(int f_num){
    if(f_num % SAMPLES_IN_EACH_FRAME != 0 ){
        /*  
        WarnLog("The frame_num of RawData %d is not a mutiple of \
                SAMPLES_IN_EACH_FRAME(%d)\n",
                f_num,SAMPLES_IN_EACH_FRAME);
                */
    }
    this->frame_num = f_num /
        SAMPLES_IN_EACH_FRAME *
        SAMPLES_IN_EACH_FRAME;
    return true;
}
