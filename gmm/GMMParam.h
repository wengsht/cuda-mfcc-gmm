/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : GMMParam.h
*D
*D 项目名称            : 
*D
*D 版本号              : 1.1.0004
*D
*D 文件描述            :
*D
*D
*D 文件修改记录
*D ------------------------------------------------------------------------------ 
*D 版本号       修改日期       修改人     改动内容
*D ------------------------------------------------------------------------------ 
*D 1.1.0001     2007.02.13     plu        创建文件
*D 1.1.0002     2007.03.20     plu        增加 m_fpLog，SetLogFile(char *p_pcLogFile)
*D 1.1.0003     2007.09.11     plu        增加MFCC类型码检查，及将该类型码写入GMM param文件
*D 1.1.0004     2007.09.17     plu        改变内存分配方式，所有的均值放在一块，所有的方差放在一块
*D 1.1.0004     2007.09.17     plu        修改了结构体GaussMixModel
*D*******************************************************************************/
#ifndef _GMMPARAM_20070213_H_
#define _GMMPARAM_20070213_H_

#include "memory_srlr.h"
#include "comm_srlr.h"
#include "Feat_comm.h"

#define LZERO (-1.0E10)			// 2007.03.20 plu : add

struct GaussPDF					// 高斯概率密度
{	
	float	*pfMean;			// 均值矢量，不分配内存，只是保存每个高斯分量均值矢量的首地址
	float	*pfDiagCov;			// 对角逆斜方差阵（矢量），内存同上
//	double	dMat;				// = log(weight) - 0.5*Dim*log(2pi) + 0.5*log|逆斜方差阵|
    float dMat;
	int		nDim;				// 有效维数（不一定是4的倍数）

	GaussPDF()
	{
		pfMean=pfDiagCov=NULL;
		dMat = 0.0;
		nDim = 0;
	}
};

struct GaussMixModel			// 混合高斯模型
{
    float		*pfWeight;		// 权重矢量 
	float		*pfMeanBuf;		// 2007.09.17 plu : 所有的均值
	float		*pfDiagCovBuf;	// 2007.09.17 plu : 所有的方差
    GaussPDF	*pGauss;		// Gaussian component
	int			nMixNum;		// 混合高斯的数目

	GaussMixModel()
	{
		pfWeight= NULL;
		pGauss  = NULL;
		nMixNum = 0;

		pfMeanBuf = pfDiagCovBuf = NULL;	// 2007.09.17 plu : 
	}
};

struct GMMFileHeader			// 混合高斯模型文件的文件头
{
	int nModelNum;				// 混合高斯模型的数目
	int nMixNum;				// 混合高斯的数目
	int nDim;					// 维数
	int nMfccKind;				// 2007.09.11 plu : 特征类型

	GMMFileHeader()
	{
		nModelNum = nMixNum = nDim = nMfccKind = -1 ; // 表示无效
	}
};

/*********************************CLASS*COMMENT***********************************
*C
*C 类名称              : GMMParam
*C
*C 类描述              : 
*C
*C
*C 类修改记录
*C ------------------------------------------------------------------------------ 
*C 修改日期       修改人     改动内容
*C ------------------------------------------------------------------------------ 
*C 2007.02.13     plu        创建类
*C*******************************************************************************/
class GMMParam 
{
protected:
	
	int		m_nModelNum;		// 混合高斯模型的数目
	int		m_nMixNum;			// 混合高斯分量的数目
	int		m_nVecSize;			// 特征维数
	int		m_nVecSize4;		// 特征维数的扩展（4的倍数）

	short   m_nMfccKind;			// 2007.09.11 plu : 保存输入的MFCC类型

	GaussMixModel *m_pGmmModel;		// 模型参数

	FILE    *m_fpLog;				// 2007.02.15 plu : Log file pointer 

protected:
	void FreeGaussMixModel(GaussMixModel *p_gmmParam,int p_nModelNum );	
	void AllocGaussMixModel(GaussMixModel *p_pModel,int p_nMixNum,int p_nVecSize);	
    
public:

	GMMParam(void);
	~GMMParam();

    void report();
    
	virtual void LoadModel(char *p_pcModelFile);		// 载入混合高斯模型
	virtual bool WriteModel(char *p_pcModelFile);		// 将模型参数写入文件

	int	GetModelNum(void)   { return m_nModelNum; };
	int GetMixtureNum(void) { return m_nMixNum; };
	int GetVecSize(void)	{ return m_nVecSize; };
    GaussMixModel * GetRawMixModel() { return m_pGmmModel; }

	void SetLogFile(char *p_pcLogFile);
    
    void LoadModel(const GMMParam &gmmParam);
    void WriteModel(const GMMParam &gmmParam);

    void AllocAll(int p_nMixNum, int p_nVecSize);
    void FreeAll();
    void InitModel(struct Features &features, int vecSize, int vecSize4);
};


#endif // _GMMPARAM_H_
