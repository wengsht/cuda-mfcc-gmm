/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : memory_srlr.cpp
*D
*D 项目名称            : 
*D
*D 版本号              : 1.1.0001
*D
*D 文件描述            :Memory Management Modules
*D
*D
*D 文件修改记录
*D ------------------------------------------------------------------------------ 
*D 版本号       修改日期       修改人     改动内容
*D ------------------------------------------------------------------------------ 
*D 1.1.0001     2007.01.01     plu        修改文件
*D*******************************************************************************/

#include "memory_srlr.h"

#include <memory.h>
#include <assert.h>

#include "comm_srlr.h"

// 分配sz个字节
void *Malloc(int sz,bool clear) {
	ASSERT3(sz>0,"Error call Malloc(int sz,const bool clear) : sz=%d!",sz);	// 2007.02.12 plu : add

	void *pt=malloc(sz); 
	if (pt==NULL)
	{ 
	   printf ("Memory can't allocate %d bytes!\a\n", sz);					// 2007.02.12 plu : add
	   char256 str;
	   sprintf (str, "Memory allocate %d bytes", sz);
	   exit(-1);
	}

	if (clear)
	{
		memset(pt,0,sz);
	}
	
	return pt;
}

// 分配p_nBlockNum*p_nSize个字节
void *Malloc(int p_nBlockNum,int p_nBlockSize,bool clear)
{
	ASSERT3(p_nBlockNum*p_nBlockSize>0,
		"Error call Malloc(int p_nBlockNum,int p_nBlockSize, const bool clear) : p_nBlockNum*p_nBlockSize=%d!",
		p_nBlockNum*p_nBlockSize);	// 2007.02.12 plu : add
	
	return Malloc(p_nBlockNum*p_nBlockSize,clear);
}

// 分配[p_nRow][p_nCol*p_nBlockSize]个字节
void **Malloc(int p_nRow,int p_nCol,int p_nBlockSize,bool clear)
{
   void **buf=(void **)Malloc(sizeof(void *)*p_nRow,clear);

   for (int i=0;i<p_nRow;i++) 
	   buf[i]=Malloc(p_nCol,p_nBlockSize,clear);
   
   return buf;
}

// 分配[p_nRow1][p_nRow2][p_nCol*p_nBlockSize]个字节
void ***Malloc(int p_nRow1,int p_nRow2,int p_nCol,int p_nBlockSize,bool clear)
{
	void ***buf=(void ***)Malloc(sizeof(void **)*p_nRow1,clear);
	for (int i=0;i<p_nRow1;i++)
	{
		buf[i]=(void **)Malloc(sizeof(void *)*p_nRow2,clear);

		for (int j=0;j<p_nRow2;j++)
			buf[i][j]=Malloc(p_nCol*p_nBlockSize,clear);
	}
	return buf;
}


void *Malloc32(int sz,bool clear)
{
   char *ptr = (char*)Malloc(sz+32,clear);
   void *pt = (void*)(ptr + 32 - ((ptr - (char*)NULL) & 31));
   ((void**)pt)[-1]=ptr;

   return pt;
}

void *Malloc32(int n1,int itsz,bool clear)
{
   return Malloc32(n1*itsz,clear);
}

void Free(void *ptr)
{ 
	if (ptr!=NULL)					// 2007.02.12 plu : 增加if(ptr) 判断
	{
		free(ptr);
		ptr = NULL;
	}
}

void Free32(void *ptr) 
{ 
	free(((void**)ptr)[-1]); 
}

void Free(void **ptr,int n1)
{
	if (ptr)					// 2007.02.12 plu : 增加if(ptr) 判断
	{
		if (n1<=0)
			printf("Warning:  Free(void **ptr,int n1) n1=%d!\a\n",n1);		// 2007.02.12 plu : add

		for (int i=0;i<n1;i++) 
			Free(ptr[i]);

		Free(ptr);
	}
}

void Free(void ***ptr,int n1,int n2)
{
	if (ptr)					// 2007.02.12 plu : 增加if(ptr) 判断
	{
		if (n1<=0)
			WARNING2("Error call Free(void **ptr,int n1,int n2) : n1=%d!",n1);		// 2007.02.12 plu : add
		if (n2<=0)
			WARNING2("Error call Free(void **ptr,int n1,int n2) : n2=%d!",n2);		// 2007.02.12 plu : add
		for (int i=0;i<n1;i++)
		{
			for (int j=0;j<n2;j++) Free(ptr[i][j]);
			Free(ptr[i]);
		}
		Free(ptr);
   }
}

