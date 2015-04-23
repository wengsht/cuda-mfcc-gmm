//
//  TimeCapture.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/27/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//
#include "AutoCapture.h"


static int patestCallback( const void *inputBuffer, void *outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void *userData )
{
    EPAnalysis * ep = (EPAnalysis*)userData;
    
    const SOUND_DATA *rptr = (const SOUND_DATA*)inputBuffer;

    bool flag = ep->addOneBlockDataWithEndFlag(rptr);
    
    if(flag == false) {
        return paComplete;
    }
    
    return paContinue;
}


bool AutoCapture::captureAction(RawData * data){
    PaError err;
    int now = 0;
    
    while( ( err = Pa_IsStreamActive( stream ) ) == 1 )
    {
        Pa_Sleep(100);
        if(PRINT_DEBUG_FRAME_INF)
            now = ep->printInf(now);fflush(stdout);
    }
    if(PRINT_DEBUG_FRAME_INF)
        now = ep->printInf(now);

    return true;
}


bool AutoCapture::init_callback(RawData * data,bool input){
    if(input){
        this->callback =patestCallback;
        this->ep->Initial(data);
        this->userData = ep;
    }
    else{
        this->callback = NULL;
    }
    return true;
}

