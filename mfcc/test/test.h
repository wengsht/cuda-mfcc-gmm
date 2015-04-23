//
//  test.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 8/26/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__test__
#define __SpeechRecongnitionSystem__test__

#include <iostream>
#include "EPAnalysis.h"
void createNoise(const char * file_name);

void capture(const char *save_file_name,
             const char *file_name,
             EPAnalysis& ep,
             bool playback = true);


SP_RESULT load_calc(const char *load_file_name,
               const char *file_name,
               EPAnalysis& ep,
               bool playback = true);

void capture(const char *save_file_name, RawData &data, bool playback);

SP_RESULT load_wav_file(const char *file_name, RawData &data);

#endif /* defined(__SpeechRecongnitionSystem__test__) */

