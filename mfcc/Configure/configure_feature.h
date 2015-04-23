//
//  configure_feature.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef SpeechRecongnitionSystem_configure_feature_h
#define SpeechRecongnitionSystem_configure_feature_h

const int SEGMENT_TIME = 25;
const int OVERLAP_TIME = 15;
const int SEGMENT_SIZE = SEGMENT_TIME/1000.0 * SAMPLES_IN_EACH_FRAME;
const int OVERLAP_SIZE = OVERLAP_TIME/1000.0 * SAMPLES_IN_EACH_FRAME;

const double SP_PREEMPH_FACTOR = 0.95;
const double WINTIME = 0.025; // sec
const double STEPTIME = 0.010;
const int    MEL_FILTER_NUM = 40;
const double MIN_F = 0.0;
const double MAX_F = 4000.0; //(SAMPLE_RATE / 2.0);
const int    CEPS_NUM = 13;
// so the feature will be 13*(2+1)
const int    DELTA_TIMES = 2;
const bool   MEL_WTS_EXIST = false;

#define DATA_PRECISION 12
#endif
