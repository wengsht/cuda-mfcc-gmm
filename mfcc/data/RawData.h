//
//  RawData.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/26/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__RawData__
#define __SpeechRecongnitionSystem__RawData__

#include "resource.h"

class RawData{
    
    // now how many frame in the main buffer
    // should always be a mutiple of FRAMES_IN_EACH_BLOCK_FOR_CAPTURE
    READ_ONLY_DECLARE(int,frame_num,FrameNum);
    
protected:
    // the main buffer , have a size of MAX_TOTAL_FRAMES
    SOUND_DATA * data;


public:
    RawData();
    RawData(const RawData& rawData);
    ~RawData();
    void clean();
    
    //
    bool setFrameNum(int f_num);
    const SOUND_DATA * getData()const;
    
    // 
    void setData(int index ,SOUND_DATA d);
    
    // using in callback
    bool appendBlockData(const SOUND_DATA * src);
    void copyBlockData(int from,int to);
    double getBlockAveEnergy(int index);

    //Save or load
    //bool saveMatlab(const char * file_name);
    bool saveWav(const char * file_name);
    bool loadWav(const char * file_name);
    //bool saveRaw(const char * file_name);
    //bool loadRaw(const char * file_name);
    
};

#endif /* defined(__SpeechRecongnitionSystem__RawData__) */
