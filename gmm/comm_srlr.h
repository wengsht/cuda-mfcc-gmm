/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : comm_GMM.h
*D
*D 项目名称            : 
*D
*D 版本号              : 1.1.0002
*D
*D 文件描述            :
*D
*D
*D 文件修改记录
*D ------------------------------------------------------------------------------ 
*D 版本号       修改日期       修改人     改动内容
*D ------------------------------------------------------------------------------ 
*D 1.1.0001     2007.02.12     plu        创建文件
*D 1.1.0002     2007.03.20     plu        ipp不包含到本头文件中
*D*******************************************************************************/
#ifndef __COMMON_GMM_H_
#define __COMMON_GMM_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include <malloc.h>
#include <memory>
#include <math.h>

#ifdef WIN32
//	#include <windows.h>
	#include <process.h>
	#include <direct.h>
	#include <io.h>

	#define getuid()	0
	#define gethostname(str,len)  DWORD buffer_name_len=len; GetComputerName(str, &buffer_name_len)
	#define sleep(n) Sleep(n)
	#define PACK_ATTRIBUTE
#else
	#include <unistd.h>
	#include <stdio.h>

	#define _MAX_FNAME			FILENAME_MAX
	#define  HANDLE				int
	#define PACK_ATTRIBUTE  __attribute__ ((packed))
#endif

#ifdef __unix__
	#define min(X,Y)  ((X) < (Y) ? (X):(Y))
	#define max(X,Y)  ((X) > (Y) ? (X):(Y))
	#define CouldBeReadOpen(X)  access(X,R_OK) >= 0
#else
	#define max(a,b)  (((a) > (b)) ? (a) : (b))
	#define min(a,b)  (((a) < (b)) ? (a) : (b))
	#define CouldBeReadOpen(X)  _access(X,4) >= 0
	#define open   _open
#endif

//#ifndef WAVE_FORMAT_PCM
	#define WAVE_FORMAT_PCM		1
	#define WAVE_FORMAT_MULAW	0x0007 
	#define WAVE_FORMAT_ALAW	0x0006 
//#endif

#define SAMPLE_8K			8000
#define SAMPLE_16K			16000

#define FILE_HEADER_SIZE	512		// 文件头的大小，参数文件的文件头均为512个字节

#define PROBTYPE			float
#define Epsilon             1.0e-5
#define Min_Mix_Weight		1.0e-5    //  log(1.0e-5) = MINMIX  in logmath.h

#define ALIGN_32(n)			(((n)+31)&(~31))
#define ALIGN_4F(n)			(((n)+3)&(~3))

#ifndef PI
	#define PI 3.14159265428
#endif

#define e 2.718281828459

#ifndef TPI
	#define TPI (2.0*PI)
#endif

const float log2pi=(float)log(TPI);	

#ifndef LZERO
	#define LZERO (-1.0E10)			// 2007.03.20 plu : add
#endif

///////////////////////////////////////////////////////////////////////////

typedef unsigned short		ushort;
typedef unsigned int		uint;
typedef unsigned char		uchar;
typedef			 char		char32[32];
typedef			 char		char64[64];
typedef			 char		char128[128];
typedef			 char		char256[256];
typedef			 char       char512[512];
typedef			 char		char1024[1024];


///////////////////////////////////////////////////////////////////////////

#define ASSERT2(CND,MSG) if (!(CND)) {fprintf(stderr, \
								"\007ERROR(%s:%d)\n" #MSG "\n",__FILE__,__LINE__ );exit(3);}
#define ASSERT3(CND,MSG,PTM) if (!(CND)) {fprintf(stderr, \
								"\007ERROR(%s:%d)\n" #MSG "\n",__FILE__,__LINE__ ,PTM);exit(3);}
#define ASSERT4(CND,MSG,PTM,PTM1) if (!(CND)) {fprintf(stderr, \
								"\007ERROR(%s:%d)\n" #MSG "\n",__FILE__,__LINE__ ,PTM,PTM1);exit(3);}
#define ASSERT5(CND,MSG,PTM,PTM1,PTM2) if (!(CND)) {fprintf(stderr, \
								"\007ERROR(%s:%d)\n" #MSG "\n",__FILE__,__LINE__ ,PTM,PTM1,PTM2);exit(3);}
#define ASSERT6(CND,MSG,PTM,PTM1,PTM2,PTM3) if (!(CND)) {fprintf(stderr, \
								"\007ERROR(%s:%d)\n" #MSG "\n",__FILE__,__LINE__ ,PTM,PTM1,PTM2,PTM3);exit(3);}

#define WARNING1(MSG) fprintf(stderr, \
								"WARNING(pid=%d,%s:%d)." #MSG "\n",getpid(),__FILE__,__LINE__ )
#define WARNING2(MSG,PTM) fprintf(stderr, \
								"WARNING(pid=%d,%s:%d)." #MSG "\n",getpid(),__FILE__,__LINE__ ,PTM)
#define WARNING3(MSG,PTM,PTM1) fprintf(stderr, \
								"WARNING(pid=%d,%s:%d)." #MSG "\n",getpid(),__FILE__,__LINE__ ,PTM,PTM1)
#define WARNING4(MSG,PTM,PTM1,PTM2) fprintf(stderr, \
								"WARNING(pid=%d,%s:%d)." #MSG "\n",getpid(),__FILE__,__LINE__ ,PTM,PTM1,PTM2)
#define WARNING5(MSG,PTM,PTM1,PTM2,PTM3) fprintf(stderr, \
								"WARNING(pid=%d,%s:%d)." #MSG "\n",getpid(),__FILE__,__LINE__ ,PTM,PTM1,PTM2,PTM3)

///////////////////////////////////////////////////////////////////////////

#define ReadOpen(X,Y)			ASSERT3(X=fopen(Y,"rb"),"Cannot open %s",Y)
#define WriteOpen(X,Y)			ASSERT3(X=fopen(Y,"wb"),"Cannot open %s",Y)
#define AppendOpen(X,Y)			ASSERT3(X=fopen(Y,"ab"),"Cannot open %s",Y)
#define TextReadOpen(X,Y)		ASSERT3(X=fopen(Y,"rt"),"Cannot open %s",Y)
#define TextWriteOpen(X,Y)		ASSERT3(X=fopen(Y,"wt"),"Cannot open %s",Y)
#define TextAppendOpen(X,Y)		ASSERT3(X=fopen(Y,"at"),"Cannot open %s",Y)

void AddWavHead(FILE *wf,long fl,int p_nWavType,int p_nSampleRate);
bool CheckWavHeader(FILE *p_fpWav,int &p_nSmpRate,int &p_nSmpNum,short &p_nWavType);

bool   CheckPath(char *inPath,bool bCreate);
bool   CheckFile(char *inFile);

bool ReadLine(FILE *p_fpTxt,char *&p_cLine);
bool GetFileName(char *&p_pcLine);

float  LogAdd(const float x,const float y);
float  LogAdd(const float x,const float y);
double LogSub(const double x,const double y);
double LogSub(const double x,const double y);

#endif
