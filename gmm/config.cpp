/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : config.cpp
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
*D 1.1.0001                                创建文件
*D 1.1.0002     2007.08.29     plu         将ReadConfig的返回类型改为bool
*D*******************************************************************************/
#include "comm_srlr.h"
#include "config.h"

//LJ-- static void SetEnv(aConfigEnv* &envs,char *config) {
void SetEnv(aConfigEnv* &envs,char *config) 
{
   char *pt;
   while ((pt=strchr(config,'"'))) strcpy(pt,pt+1);
   if ((pt=strchr(config,'='))) {
      aConfigEnv *tmp=new aConfigEnv;
      tmp->next=envs; envs=tmp; 
      *pt='\0'; strcpy(tmp->env,config); 
      strcpy(tmp->def,pt+1); *pt='=';
   }
}

Config::Config()
{
	fenv=NULL;
	exam=false;
	envs=NULL;
}

Config::~Config(void)
{
	if (envs!=NULL)
	{
		for (aConfigEnv *tenv,*env=envs;(tenv=env);delete tenv)
			env=env->next;
	}

	if (exam)	exit(3);

	if (fenv!=NULL) 
	{
		fprintf(fenv,"Compiling date: %s\n",__DATE__);
		fclose(fenv);
	}
}

// 2007.08.29 plu : add
bool Config::SetConfigFile(char *cfgFile)
{
	FILE *fin;
	fin=fopen(cfgFile,"rt");
	if (fin==NULL)		
	{
		printf("Error open %s for read!\n",cfgFile);
		return false;
	}
	
	char256 config="";
    while (!feof(fin)) 
	{
		if (fscanf(fin,"\n %[^\n]s",config)<1) break;
        SetEnv(envs,config);
    }
    fclose(fin);

	return true;
}

char *Config::GetEnv(const char *env) 
{
   for (aConfigEnv *ev=envs;ev;ev=ev->next)
      if (!strcmp(ev->env,env)) return ev->def;
   char *tmp=getenv(env);
   if (tmp==NULL)
	   printf("parameter %s not set\n",env);
   return tmp;
}

bool Config::ReadConfig(const char *line,int& num) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	sscanf(pRst,"%d",&num);
	if (fenv) fprintf(fenv,"%s=%d\n",line,num);

	return true;
}

bool Config::ReadConfig(const char *line,int& num1,int& num2) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	sscanf(pRst,"%d %d",&num1,&num2);
	if (fenv) fprintf(fenv,"%s=%d %d\n",line,num1,num2);

	return true;
}

bool Config::ReadConfig(const char *line,bool& bln) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	bln=(!strncmp(pRst,"true",4));
	if (fenv) fprintf(fenv,"%s=%s\n",line,(bln)?"true":"false");

	return true;
}

bool Config::ReadConfig(const char *line,float& num) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	sscanf(pRst,"%g",&num);
	if (fenv) fprintf(fenv,"%s=%g\n",line,num);

	return true;
}

bool Config::ReadConfig(const char *line,float& num1,float& num2) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	sscanf(pRst,"%g %g",&num1,&num2);
	if (fenv) fprintf(fenv,"%s=%g %g\n",line,num1,num2);

	return true;
}

bool Config::ReadConfig(const char *line,float& num1,float& num2,float& num3) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	sscanf(pRst,"%g %g %g",&num1,&num2,&num3);
	if (fenv) fprintf(fenv,"%s=%g %g %g\n",line,num1,num2,num3);

	return true;
}

bool Config::ReadConfig(const char *line,float& n1,float& n2,float& n3,float& n4) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	if (sscanf(pRst,"%g ,%g %g ,%g",&n1,&n2,&n3,&n4)<3) { n3=n1;n4=n2; }
	if (fenv) fprintf(fenv,"%s=%g,%g %g,%g\n",line,n1,n2,n3,n4);

	return true;
}

bool Config::ReadConfig(const char *line,char *str) 
{
    char buf[256];
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	str[0]='\0'; 
    sscanf(pRst,"%s",str);

	if (fenv) fprintf(fenv,"%s=%s\n",line,str);

	return true;
}

bool Config::ReadConfig(const char *line,char *str1,char *str2) 
{
	char *pRst = GetEnv(line);
	if (pRst==NULL)	return false;

	if (sscanf(pRst,"%s %s",str1,str2)<2) strcpy(str2,str1);
	if (!strcmp(str1,"null") || !strcmp(str1,"NULL")) str1[0]='\0';
	if (!strcmp(str2,"null") || !strcmp(str2,"NULL")) str2[0]='\0';
	if (fenv) fprintf(fenv,"%s=%s %s\n",line,
	  (str1[0])?str1:"null",(str2[0])?str2:"null");

	return true;
}
