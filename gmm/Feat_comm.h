/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : Feat_comm.h
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
*D 1.1.0001     2007.12.24     plu        创建文件
*D 1.1.0002     2007.12.25     plu        支持SDC
*D*******************************************************************************/

#ifndef _FEAT_COMM_20071224_H_
#define _FEAT_COMM_20071224_H_

enum _BaseFeatureKind			// 特征基本类型，注意不能超过7
{
	FYT_MFCC=1,
	FYT_MFCCPLP,
	FYT_RASTAPLP
};

#define KINDMASK    (0x3f)

#define DYN_RANGE   50

// 编码定义，8进制，转换到2进制时，只有一位是1
#define HASENERGY    010       // _E log energy included , 
#define HASZEROC     020       // _0 0'th Cepstra included , 注意：_E _0 不能同时设置
#define HASDELTA     040       // _D delta coef appended ，1阶差分
#define HASACCS     0100       // _A acceleration coefs appended ，2阶差分
#define HASTHIRD    0200       // _T has Delta-Delta-Delta index attached ，3阶差分

#define HASSDC	  010000	   // _S SDC 差分

// 规整技术定义
#define DOCMN		0400		
#define DOCVN	   01000       
#define DOWARP	   02000
#define DORASTA    04000

// 注： 因为m_MFCCKind的类型是short型，还可以用的码有 
//   010000						// 2007.12.25 plu : sdc 占用了
//   020000
//   040000
//  0100000
				   
#define BASEMASK  07         // 用来回复出m_MFCCKind最基本的类型，FYT_MFCC、FYT_MFCCPLP、FYT_RASTAPLP

struct FEATURE_MFCCBASEINFO  // 特征类所需的基本信息
{
    char  targetKind[16];
    
    int   smpPeriod;				// 采样周期，帧长
    int   framePeriod;
    
    // MfccBased
    bool  zeroGlobalMean;
	int   chlNum;					// 滤波器数目（通道数），倒谱数
	int	  cepNum;

	int	  cepLifter;				// 倒谱提升，窗长
	int	  winSize;
		    
	float lowPass;					// 低/高截至频率
	float highPass;
	
	// EnergyBased
	bool  normEnergy;
	float energyScale;
	float silFloor;
		
	// robust Processing
	bool  doCMN;
	bool  doCVN;
	bool  doRASTA;
	float RASTACoff;
	bool  doFeatWarp;

	// SDC相关参数
	int	  nSdc_N;		// 7
	int   nSdc_D;		// 1
	int	  nSdc_P;		// 3
	int   nSdc_K;		// 7
	

	FEATURE_MFCCBASEINFO()
	{
		targetKind[0]='\0';
    
		smpPeriod=625;					// 采样周期，帧长
		framePeriod=100000;
    
		// MfccBased
		zeroGlobalMean=false;
		chlNum=24;						// 滤波器数目（通道数），倒谱数
		cepNum=12;

		cepLifter=22;					// 倒谱提升，窗长
		winSize=250000;
		
		lowPass=-1.f;					// 低/高截至频率
		highPass=-1.f;
			
		// EnergyBased
		normEnergy=false;
		energyScale=1.f;
		silFloor=50.f;

		// robust processing
		doCMN = false;
		doCVN = false;
		doRASTA = false;
		doFeatWarp = false;
		RASTACoff = 0.98f;	// or 0.94

		// SDC
		nSdc_N=7;		// 7
		nSdc_D=1;		// 1
		nSdc_P=3;		// 3
		nSdc_K=7;		// 7
	};
};


struct Feature_BaseInfo		// 特征文件头里的保存的信息
{
	char cFeatType[16];		// 例如：MFCC_A
	int  nFrameNum;			// 帧数
	int  nVecSize;			// 特征维数
	int  nVecSizeStore;		// 实际存储的特征维数
	int  nFeatKind;			// 特征类型
	int  nWinSize;			// 短时窗的窗长
	int  nFrameRate;		// 帧率

	int  nTotalParamSize;	// 特征总的参数量，单位：字节

	Feature_BaseInfo()
	{
		cFeatType[0]='\0';
		nFrameNum = nVecSizeStore = nVecSize = nFeatKind = -1;
		nFrameRate = nWinSize = -1;
		nTotalParamSize = 0;
	}
};

struct Features {
    float ** features;
    int nFeatures;
    int * featureSize;
    int featureDim;
};

void ReadFeatures(char *p_pcFlistFile, struct Features &features, int &maxFrameNum, int VecDim);
void ReadFeature(char *p_pcFeatFile, float **feature, int *featureSize);
void FreeFeatures(struct Features &features);

bool ReadFeatFile(char *p_pcFeatFile,float *&p_pfFeatBuf,Feature_BaseInfo &p_sFeatInfo);
bool WriteFeatFile(char *p_pcFeatFile,float *p_pfFeatBuf,Feature_BaseInfo &p_sFeatInfo);
bool IsSameFeat(Feature_BaseInfo &p_sFeatInfo1,Feature_BaseInfo &p_sFeatInfo2);
void CopyFeatBaseInfo(Feature_BaseInfo &p_sDstInfo,Feature_BaseInfo &p_sSrcInfo);
bool ReadLimitFeatFile(char *p_pcFeatFile,float *&p_pfFeatBuf,Feature_BaseInfo &p_sFeatInfo, int nLimitLeng);
#endif // _FEAT_COMM_H_
