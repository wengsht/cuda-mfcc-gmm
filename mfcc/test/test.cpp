#include "srs.h"
#include <stdlib.h>
#include "test.h"
#include <termios.h>
#include <unistd.h>
#include <string.h>
//#include <direct.h>
#include <stdio.h>

void createNoise(const char * file_name){
    RawData data;
    int size =SAMPLES_IN_EACH_FRAME * FRAME_PER_SECOND * 60;
    data.setFrameNum(size);
    
    for(int i = 0;i<size;i++){
        int d = rand()%500+10000;
        data.setData(i,d);
    }
    data.saveWav(file_name);
}

const char * stringFile(const char * a,const char * b,char *ab){
    ab[0] = '\0';
    strcpy(ab,SAVE_DATA_DIR);
    strcat(ab,a);
    strcat(ab,b);
    return ab;
}

void capture(const char * save_file_name,
             const char *file_name,
             EPAnalysis& ep,
             bool playback){
    RawData data;
    AutoCapture c(&ep);
    
    Tip("Press any key to capture:");
    getch();
    puts("");
    Tip("Preparing...");
    sleep(1);
    
    char fn_buffer [128]="";
    
    Tip("Start talking");

    if(c.capture(&data)){
        data.saveWav(stringFile(save_file_name,".wav",fn_buffer));
        ep.saveMatlab(stringFile(file_name,".dat",fn_buffer));
        
        //if(playback)c.play(&data);
        
        ep.smooth();
        ep.saveMatlab(stringFile(file_name,"_smooth.dat",fn_buffer));
        ep.cut();
        ep.saveMatlab(stringFile(file_name,"_cut.dat",fn_buffer));
        
        if(playback) c.play(&data);
        
        data.saveWav(stringFile(save_file_name,"_cut.wav",fn_buffer));

    }
    else{
        ErrorLog("Capture error");
    }
}

void capture(const char *save_file_name, RawData &data, bool playback) {
    DAEPAnalysis ep;
    AutoCapture c(&ep);
    
    Tip("Press any key to capture:");
    getch();
    puts("");
    Tip("Preparing...");
    sleep(1);
    
    char fn_buffer [128]="";
    
    Tip("Start talking");

    if(c.capture(&data)){
        data.saveWav(stringFile(save_file_name,".wav",fn_buffer));
        
        //if(playback)c.play(&data);
        
        ep.smooth();
        ep.cut();
        
        if(playback) c.play(&data);
        
        data.saveWav(stringFile(save_file_name,"_cut.wav",fn_buffer));
    }
    else{
        ErrorLog("Capture error");
    }
}


SP_RESULT load_calc(const char *load_file_name,
               const char *file_name,
               EPAnalysis& ep,
               bool playback) {
    RawData data;
    Capture c;
    char fn_buffer [128]="";
    
    data.loadWav(stringFile(load_file_name,".wav",fn_buffer));
    ep.Initial(&data);
    ep.reCalcAllData();
    ep.saveMatlab(stringFile(file_name,".dat",fn_buffer));
    
    if(playback)c.play(&data);
    
    ep.smooth();
    ep.saveMatlab(stringFile(file_name,"_smooth.dat",fn_buffer));
    ep.cut();
    ep.saveMatlab(stringFile(file_name,"_cut.dat",fn_buffer));
    
    //ep.smooth();
    //ep.cut();
    //ep.saveMatlab(stringFile(file_name,"_cut_2.m",fn_buffer));

    if(playback)c.play(&data);

    return SP_SUCCESS;
}

SP_RESULT load_wav_file(const char *file_name, RawData &data) {
    DAEPAnalysis da_ep;
    Capture c;
    char fn_buffer[128] = "";
    data.loadWav(stringFile(file_name, ".wav", fn_buffer));
    da_ep.Initial(&data);
    da_ep.reCalcAllData();
//    da_ep.smooth();
//    da_ep.cut();

    return SP_SUCCESS;
}
