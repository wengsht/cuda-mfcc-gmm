//
//  tool.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/7/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef SpeechRecongnitionSystem_tool_h
#define SpeechRecongnitionSystem_tool_h

#include <stdio.h>
#include <assert.h>
#include <stdarg.h>
#include <ctype.h>
#include "tool.h"

///
/// Debug Tool
///

const int TESTING = true;

void Tip(const char * msg,...);
void Warn(const char * msg,...);
void Log(const char * msg,...);
void Log(const char *filename, const int line_no, const char * msg,...);
void ErrorLog(const char *msg,...);
void WarnLog(const char *msg,...);

///
/// IO helper
///

int getch();
int getche();

////////////////////////////////////////////////////FILE
void saveArray(FILE* fid ,const double * data, int len);
void saveArray(FILE* fid ,const char * data, int len);
void saveArray(FILE* fid ,const int * data, int len);

///
/// some tool, which is nothing to do with the algorithm logic
///
#define NONE         "\033[m"
#define RED          "\033[0;32;31m"
#define LIGHT_RED    "\033[1;31m"
#define GREEN        "\033[0;32;32m"
#define LIGHT_GREEN  "\033[1;32m"
#define BLUE         "\033[0;32;34m"
#define LIGHT_BLUE   "\033[1;34m"
#define DARY_GRAY    "\033[1;30m"
#define CYAN         "\033[0;36m"
#define LIGHT_CYAN   "\033[1;36m"
#define PURPLE       "\033[0;35m"
#define LIGHT_PURPLE "\033[1;35m"
#define BROWN        "\033[0;33m"
#define YELLOW       "\033[1;33m"
#define LIGHT_GRAY   "\033[0;37m"
#define WHITE        "\033[1;37m"

#define RED_BACK "\033[7;31m"
#define GREEN_BACK "\033[7;32m"
#define BLUE_BACK "\033[7;34m"

#define NONE_BACK "\033[0m"


#define READ_ONLY_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public : virtual TYPE get##FUNC()const{return NAME;}

#define CONST_REFERENCE_READ_ONLY_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public : virtual const TYPE & get##FUNC()const{return NAME;}

#define REFERENCE_READ_ONLY_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public : virtual TYPE & get##FUNC() {return NAME;}


#define WRITE_ONLY_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public :virtual void set##FUNC(TYPE name){this->NAME = name;}

#define READ_WRITE_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public :virtual TYPE get##FUNC()const{return NAME;} \
public :virtual void set##FUNC(TYPE name){this->NAME = name;}

#define READ_INIT_DECLARE(TYPE,NAME,FUNC) \
protected  :TYPE NAME; \
public :virtual TYPE get##FUNC()const{return NAME;} \
public :virtual void init##FUNC(TYPE name){this->NAME = name;}

#define SAFE_DELETE_POINTER(val) \
do{if(val)delete val;}while(0)

#define ERROR_CODE(DUMMY1, DUMMY2, DUMMY3) \
    +1
const int  SP_RESULT_CNT = 1
#include "ErrorCode.def"
;
#undef ERROR_CODE
enum ERROR_UIDS {
    SP_SUCCESS
#define ERROR_CODE(ERROR_UID, DUMMY1, DUMMY2) \
    , ERROR_UID
#include "ErrorCode.def"
#undef ERROR_CODE
};

extern const char *SP_ERROR_CODE_GLOBAL[];

typedef int SP_RESULT;
const char * SP_ERROR_CODE(SP_RESULT );

#endif
