//
//  Capture.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/26/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "Capture.h"

#define DITHER_FLAG     (0) /**/
///
///
///

void CHECK_ERR(PaError err){
    if( err != paNoError ) {
        ErrorLog(Pa_GetErrorText(err));
    }
}


///
///
///
Capture::Capture(){
    callback = NULL;
    userData = NULL;
}

Capture::~Capture(){
}

bool Capture::init_callback(RawData * data,bool input){
    return true;
}

bool Capture::init_PA(){
    PaError err;
    err = Pa_Initialize();
    CHECK_ERR(err);
    return true;
}


bool Capture::init_stream(bool input){
    PaError err;
    if (input){
        in.device = Pa_GetDefaultInputDevice(); /* default input device */
        if (in.device == paNoDevice){
            ErrorLog("No default input device.");
            return false;
        }
        in.sampleFormat = PA_DATA_TYPE;
        
        in.channelCount = NUM_CHANNELS;
        
        in.suggestedLatency = Pa_GetDeviceInfo( in.device )->defaultLowInputLatency;
        in.hostApiSpecificStreamInfo = NULL;
        err = Pa_OpenStream(&stream,&in,NULL,
                            SAMPLE_RATE,SAMPLES_IN_EACH_FRAME,
                            paClipOff,callback,userData );
    }
    else{
        out.device = Pa_GetDefaultOutputDevice();
        if (out.device == paNoDevice) {
            ErrorLog("No default output device.");
            return false;
        }
        out.sampleFormat = PA_DATA_TYPE;
        out.channelCount = NUM_CHANNELS;
        out.suggestedLatency = Pa_GetDeviceInfo( out.device )->defaultLowOutputLatency;
        out.hostApiSpecificStreamInfo = NULL;
        err = Pa_OpenStream(&stream,NULL,&out,
                            SAMPLE_RATE,SAMPLES_IN_EACH_FRAME,
                            paClipOff,callback,userData );
    }
    
    
    CHECK_ERR(err);

    return true;
}


bool Capture::end(){
    PaError err;
    err = Pa_CloseStream( stream );
    CHECK_ERR(err);

    err = Pa_Terminate();
    CHECK_ERR(err);

    return true;
}


bool Capture::init(RawData * data,bool input){
    Log("Initial capture");
    
    if(input)  data->clean();

    if(! this->init_callback(data, input)) return false;
    
    if(! this->init_PA())return false;
    
    if(! this->init_stream(input) )return false;
    

    PaError err = Pa_StartStream( stream );
    if( err != paNoError ) {
        ErrorLog(Pa_GetErrorText(err));
        return false;
    }
    
    Log("Initial successfully");

    return true;
}

       
bool Capture::capture(RawData * data){
    Tip("Capturing...");
    if(init(data,true)==false) return false;
    
    if(this->captureAction(data) == false) return false;
    
    if(! end())return false;
    
    return true;
    
}

bool Capture::play(RawData * data){
    Tip("Playing...");
    if(init(data,false)==false) return false;
    
    if(this->playAction(data) == false) return false;
    
    if(! end())return false;
    
    return true;
}

bool Capture::captureAction(RawData * data){
////    printf("%d\n",data->getTotalFrame());
//
//    PaError err = Pa_ReadStream( stream,
//                                data->getData(),
//                                MAX_SIZE
//                                );
//    
//    CHECK_ERR(err);
    return true;
}

bool Capture::playAction(RawData * data){
    PaError err = Pa_WriteStream( stream,
                                 data->getData(),
                                 data->getFrameNum()
                                 );
    
    
    CHECK_ERR(err);
    return true;
}

//attribute


