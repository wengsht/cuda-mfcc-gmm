//
//  Capture.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/26/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__Capture__
#define __SpeechRecongnitionSystem__Capture__

#include "RawData.h"
#include "portaudio.h"
#include "DAEPAnalysis.h"

//do the initial job ,etc. 
class Capture{
protected:
    //basic
    PaStreamParameters in;
    PaStreamParameters out;
    PaStream * stream;
    

    // initial and end
    bool init_stream(bool input );
    bool init_PA();
    
    // the basic class don't use callback
    bool init(RawData * data,bool input );
    bool end();
    
    //use in callback model;
    PaStreamCallback* callback;
    void * userData;
    
    virtual bool captureAction(RawData * data) ;
    virtual bool playAction(RawData * data) ;
    virtual bool init_callback(RawData * data,bool input);


public:
    static SP_RESULT load_wav_file(const char * const file_name, RawData &data, bool playback = false) {
        DAEPAnalysis da_ep;
        Capture c;
        char fn_buffer[128] = "";
        data.loadWav(file_name);
        da_ep.Initial(&data);
        da_ep.reCalcAllData();

        if(playback)
            c.play(&data);
        //    da_ep.smooth();
        //    da_ep.cut();

        return SP_SUCCESS;
    }
    Capture();
    ~Capture();
    // 
    bool  capture(RawData *);
    bool  play(RawData *);

};

#endif
