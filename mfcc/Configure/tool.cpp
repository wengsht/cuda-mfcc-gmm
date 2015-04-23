//
//  tool.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include <stdio.h>
#include <stdarg.h>
#include <ctype.h>
#include <termios.h>
#include <unistd.h>
#include "tool.h"
#include <stdlib.h>

void Warn(const char *msg, ...){
    //if(! TESTING) return;
	char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
    toupper(Buffer[0]);
	printf("" LIGHT_RED"WARNNING: ");
	printf ("%s", Buffer);
	printf(NONE "\n");
}

void Tip(const char * msg,...){
    char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
//	printf("Tip: ");
    toupper(Buffer[0]);
	printf ("%s", Buffer);
	printf(NONE "\n");
}

void Log(const char *filename, const int line_no, const char * msg,...){
    if(! TESTING) return;

	char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
    toupper(Buffer[0]);
	printf("Log: ");
    printf("File[%s] line[%d]: ", filename, line_no);
	printf ("%s", Buffer);
	printf(NONE "\n");
}

void Log(const char * msg,...){
    if(! TESTING) return;
	char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
    toupper(Buffer[0]);
	printf("Log: ");
	printf ("%s", Buffer);
	printf(NONE "\n");
}

void ErrorLog(const char *msg,...){
    //if(! TESTING) return;
	char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
    toupper(Buffer[0]);
	printf("\n" LIGHT_RED"### ERROR: ");
	printf ("%s", Buffer);
	printf(NONE "\n\n");
    fflush(stdin);
    getch();
}

void WarnLog(const char *msg,...){
    if(! TESTING) return;
	char Buffer[128];
	va_list ArgList;
	va_start (ArgList, msg);
	vsprintf (Buffer, msg, ArgList);
	va_end (ArgList);
    toupper(Buffer[0]);
	printf("\nWARNNING: ");
	printf ("%s", Buffer);
	printf("\n\n");
    fflush(stdin);
    getch();
}



int getch() {
    struct termios oldt,
    newt;
    int ch;
    tcgetattr( STDIN_FILENO, &oldt );
    newt = oldt;
    newt.c_lflag &= ~( ICANON | ECHO );
    tcsetattr( STDIN_FILENO, TCSANOW, &newt );
    ch = getchar();
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt );
    return ch;
}

int getche() {
    int ch = getch();
    printf("%c",ch);
    return ch;
}


////////////////////////////////////////////////////FILE
void saveArray(FILE* fid ,const double * data, int len){
    fprintf(fid,"%d\n",len);
    for(int i = 0;i<len;i++){
        fprintf(fid,"%lf ",data[i]);
    }
    fprintf(fid,"\n");
}

void saveArray(FILE* fid ,const char * data, int len){
    fprintf(fid,"%d\n",len);
    for(int i = 0;i<len;i++){
        fprintf(fid,"%d ",(int)data[i]);
    }
    fprintf(fid,"\n");
}


void saveArray(FILE* fid ,const int * data, int len){
    fprintf(fid,"%d\n",len);
    for(int i = 0;i<len;i++){
        fprintf(fid,"%d ",data[i]);
    }
    fprintf(fid,"\n");
}

const char *SP_ERROR_CODE_GLOBAL[] = {
    "SUCCESS"
#define ERROR_CODE(ERROR_UID, X, STR) \
    , STR
#include "ErrorCode.def"
#undef ERROR_CODE
};
const char * SP_ERROR_CODE(SP_RESULT ERROR_UID) {
    if(SP_RESULT_CNT <= ERROR_UID) 
        return "NULL ERROR TYPE";
    return SP_ERROR_CODE_GLOBAL[ERROR_UID];
}
