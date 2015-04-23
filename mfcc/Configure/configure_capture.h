//
//  configure_capture.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef SpeechRecongnitionSystem_configure_capture_h
#define SpeechRecongnitionSystem_configure_capture_h
#include "configure_basic.h"
/// change these configures to make the program more efficient
///


// ep1
const int  SILENCE_ENERGY_THRESHOLD  = 30;  // as lower as possible


// ep2
const double FORGET_FACTOR = 3;       // for bg , smaller and faster
const double ADJUSTMENT = 0.01;         // for level , smaller and slower
const double LEVEL_BG_THRESHOLD = 10;

// ep3


const double ONSET_THRESHOLD = 10;      //decrease in noise environment
const double OFFSET_THRESHOLD = -2;     //increase in noise environment
const double ALPHA = 0.9;               //small and slower(in speech)
const double TRACKING_FACTOR = 0.01;    // small and track quikly(in silent)

//const double ONSET_THRESHOLD = 15;      //decrease in noise environment
//const double OFFSET_THRESHOLD = -6;     //increase in noise environment
//const double ALPHA = 0.9;               //small and slower(in speech)
//const double TRACKING_FACTOR = 0.01;    // small and track quikly(in silent)




const int MAX_TOTAL_SAMPLE = MAX_BUFFER_SECOND *SAMPLE_RATE;
const int MAX_SIZE =MAX_TOTAL_SAMPLE *NUM_CHANNELS;

//
const double MIN_SILENT_TIME = 100;
const double MIN_SPEECH_TIME = 200;
const double EXTENT_SPEECH = 250;  //??? no use now

const int MIN_SILENT_FRAMES = MIN_SILENT_TIME * FRAME_PER_SECOND / 1000;
const int MIN_SPEECH_FRAMES = MIN_SPEECH_TIME * FRAME_PER_SECOND /1000;
const int EXTENT_SPEECH_FRAMES = EXTENT_SPEECH * FRAME_PER_SECOND / 1000;

#endif
