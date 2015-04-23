//
//  RawDataIO.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/6/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include <stdio.h>
#include "readwave.h"
#include "RawData.h"

/*
void draw_raw(int size,SOUND_DATA * data,FILE* fid){
    
    fprintf(fid,"x=[");
    for(int i =0;i<size;i++){
        fprintf(fid," %d",i);
    }
    fprintf(fid,"];\n");
    
    fprintf(fid,"y=[");
    for(int i =0;i<size;i++){
        fprintf(fid," " PRINTF_S_FORMAT,data[i]);
    }
    fprintf(fid,"];\n");
    
    fprintf(fid,"plot(x,y)\n");
    
}

bool RawData::saveMatlab(const char * file_name){

    FILE  *fid;
    fid = fopen(file_name, "w");
    if( fid == NULL )
    {
        printf("Could not open file: %s\n",file_name);
        return false;
    }
    else
    {
        fprintf(fid,"figure; hold on;\n");
        // draw the raw data
        draw_raw(this->frame_num,data,fid);

        
        fclose( fid );
        printf("Wrote data to: %s\n",file_name);
    }
    return true;
}
 */

bool RawData::saveWav(const char * file_name){
    Tip("Write to %s",file_name);
    WriteWave(file_name,this->data,this->frame_num,SAMPLE_RATE);
    return true;
}

bool RawData::loadWav(const char * file_name){
    Tip("Load from %s",file_name);
    int t_frame,s_rate;
    this->data = ReadWave(file_name, &t_frame,&s_rate);
    if (s_rate != SAMPLE_RATE){
        ErrorLog("Wav File has a sample_rate of %d, but now SAMPLE_RATE = %d\n"
                 ,s_rate,SAMPLE_RATE);

    }
    setFrameNum(t_frame);
    return true;
}

/*
bool RawData::loadRaw(const char * file_name){
    
    FILE  *fid;
    fid = fopen(file_name, "rb");
    if( fid == NULL )
    {
        printf("Could not open file: %s\n",file_name);
        return false;
    }
    else
    {
        int total_frame;
        fread( &total_frame, 1, sizeof(int), fid );
        
        if(setFrameNum(total_frame)==false){
            printf("Error: Raw File has a total frames of %d, but a total frames should be multiple of  %d\n"
                   ,total_frame,FRAMES_IN_EACH_BLOCK_FOR_CAPTURE);
            fclose( fid );
            return false;
        }
        
        fread( data,NUM_CHANNELS * sizeof(SOUND_DATA),total_frame, fid );
 
        fclose( fid );
        
        printf("Wrote data to: %s\n",file_name);
    }
    return true;
}

bool RawData::saveRaw(const char * file_name){
    FILE  *fid;
    fid = fopen(file_name, "wb");
    if( fid == NULL )
    {
        printf("Could not open file: %s\n",file_name);
        return false;
    }
    else
    {
        fwrite( &(this->frame_num), 1, sizeof(int), fid );
        fwrite( data,
               NUM_CHANNELS * sizeof(SOUND_DATA),
               this->frame_num, fid );
        fclose( fid );
        printf("Wrote data to: %s\n",file_name);
    }
    return true;
}*/