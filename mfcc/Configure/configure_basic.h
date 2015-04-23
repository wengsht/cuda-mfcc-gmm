//
//  configure_basic.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef SpeechRecongnitionSystem_configure_basic_h
#define SpeechRecongnitionSystem_configure_basic_h

#define WAV_SUFFIX ".wav"
#define MFCC_SUFFIX ".mfcc"
///
/// No very important configure, but don't change they for fun.
///
const double  STOP_IN_SECONDS = 1; // how many second to wait when in silent
const int MAX_BUFFER_SECOND  = 120 ; // 2 minutes, don't too small
const int FRAME_PER_SECOND = 40; //don't change


///
/// performance configure
///
const bool PRINT_DEBUG_FRAME_INF = true; // print the block energy,bg,etc.
const char  SAVE_DATA_DIR[] = "./";



///
/// don't change this configure for fun,
/// or you can't user some file that made by these configure
/// and you also need to change other configures to make the algorithm work well.
///
const int SAMPLE_RATE = 8000; // 16000; // 44100;  //don't change
const int NUM_CHANNELS = 1; //don't change

const int SAMPLES_IN_EACH_FRAME = SAMPLE_RATE / FRAME_PER_SECOND;

const int MAX_BUFFER_SIZE = SAMPLE_RATE * MAX_BUFFER_SECOND;

typedef double FEATURE_DATA;

#if 0
#define PA_DATA_TYPE  paFloat32
typedef float SOUND_DATA;
#define DATA_SILENCE  (0.0f)
#define PRINTF_S_FORMAT "%.8f"
#elif 1
#define PA_DATA_TYPE  paInt16
typedef short SOUND_DATA;
#define DATA_SILENCE  (0)
#define PRINTF_S_FORMAT "%d"
#elif 0
#define PA_DATA_TYPE  paInt8
typedef char SOUND_DATA;
#define DATA_SILENCE  (0)
#define PRINTF_S_FORMAT "%d"
#else
#define PA_DATA_TYPE  paUInt8
typedef unsigned char SAMPLE;
#define DATA_SILENCE  (128)
#define PRINTF_S_FORMAT "%d"
#endif

#include <vector>
template <class T>
using Matrix = std::vector< std::vector<T> > ;

#endif
