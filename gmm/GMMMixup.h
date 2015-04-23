/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : GMMMixup.h
*D
*D 项目名称            : 
*D
*D 版本号              : 1.1.0001
*D
*D 文件描述            : 对GMM模型进行mixture分裂
*D
*D
*D 文件修改记录
*D ------------------------------------------------------------------------------ 
*D 版本号       修改日期       修改人     改动内容
*D ------------------------------------------------------------------------------ 
*D 1.1.0001     2007.03.20     plu        创建文件
*D*******************************************************************************/

#ifndef _GMMMIXUP_20070320_H_
#define _GMMMIXUP_20070320_H_

#include "GMMParam.h"

#define MAXSPLITNUM		5000				// 2007.03.20 plu : 一个高斯分量最大允许的分裂次数
#define MINMIX			1.0E-5				/* Min usable mixture weight */
#define LMINMIX			11.5129254649702	/* log(MINMIX) */

class GMMMixup : public GMMParam
{
protected:
	float			m_fMeanGC,m_fStdGC;
	float			m_fSplitDepth;			// 分裂深度控制参数
	int				*m_pnSplitCount;		// 记录每个高斯分量被分裂的次数
	int				m_nNewMixNum;			// 分裂后的mixNum
	GaussMixModel	*m_pNewGmmModel;		// 分裂后的模型buf
	
	// 分裂用，得到高斯分量中GConst的一阶二阶统计量
	void    GConstStats(GaussMixModel *p_pGmmModle);		

	// 从所有高斯分量中选择一个最适合分裂的,p_nModelIdx为模型的index
	int     HeaviestMix(GaussMixModel *p_pGmmModel,int p_nMixNum);

    // 将某个高斯分量分裂成两个分量
	// p_pGmmModel为模型Buf指针
	// p_nSrcMixIdx 是待分裂的高斯index，
	// p_nSrcMixIdx，p_nDstMixIdx是分裂后的高斯index
    void    SplitMix(GaussMixModel *p_pGmmModel,int p_nSrcMixIdx,int p_nDstMixIdx);
	
	void    CloneMixPDF(GaussPDF &p_pDst,GaussPDF &p_pSrc);
	void    FixInvDiagGConst(GaussMixModel *p_pGmmModel);
	void    FixInvDiagMat(GaussMixModel *p_pGmmModel);
	void	CopyModel(GaussMixModel *pGMMDes,GaussMixModel *pGMMSrc);
	void    CheckWeight(GaussMixModel *p_pGmmModel);
public:

	GMMMixup();
	~GMMMixup();

	// 分裂训练
	void Mixup(int p_nNewMixNum,float p_fDepth);		// 模型分裂
};

#endif // _GMMMIXUP_H_
