#ifndef _GMMTRAIN_20070213_H_
#define _GMMTRAIN_20070213_H_

#include "Feat_comm.h"
#include "GMMParam.h"

struct GaussStatis				
{
	float *pfMeanAcc;		
	float *pfVarAcc;	
	
	double dOcc;	

	GaussStatis()
	{
		pfMeanAcc=pfVarAcc=NULL;
		dOcc=0.0;
	}
};

struct GMMStatic
{
	float	fTotalProb;
	double  dTotalOcc;			
	int		nTotalNum;		

	GaussStatis *pGauss;

	GMMStatic()
	{
		fTotalProb=0.f;
		dTotalOcc = 0.0;
		nTotalNum = 0;
		pGauss = NULL;
	}
};

class GMMTrain : public GMMParam
{
protected:

	GMMStatic   *m_pGMMStatic;		// *gAcc
	
	float	m_fWeightFloor;			// float wFloor;	weight的更新门限
	float   *m_pfVarFloor;			// float *varFloor; 方差的更新门限
	float	m_fOccFloor;			// float minocc;    高斯分量的更新门限，小于改值的高斯分量不更新

	float   m_fVarFloor;			// 2007.03.16 plu : 所有维数的Var用一个Floor

	int		m_nMaxFrameNum;			// OLD: int num  最大帧数
//	float	*m_pfFeatBuf;			// 2007.01.20 plu : 特征Buffer
	float	**m_ppfProb_BufBuf;		// float **tmpV;
	float	*m_pfProb_Buf;			// float *prob;	
	
	// 2007.09.16 plu : 为IPP计算增加的buf，大小为VecSize4
	float	*m_pfIPPBuf_VecSize4;

    struct Features *features;
    
protected:
	
    void AllocProbBuf(int m_nMaxFrameNum,int m_nMixNum);
	virtual void	FreeStaticBuf();			
	//void			ReadStatic_SumStatic(char *p_cFileName);

public:

	GMMTrain();
	~GMMTrain();

    void DataPrepare(struct Features &features, int MaxMixNum);
    
    void EMTrain(int EMIterNum);
    
	void SetFloor(float p_fOccFloor,float p_fWeightFloor,float p_fVarFloor);
	
    void AllocStaticBuf(int m_nModelNum, int m_nMixNum, int m_nVecSize4, int m_nMaxFrameNum);
    void ComputeStatistics();
    void cuda_ComputeStatistics();
	virtual void ResetStaticBuf();			
    
    void EMIteration();

	//void WriteStatic(char *p_pcFileName);	

	virtual void ComputeStatiscs_MT(float *m_pfFeatBuf, int nFrameNum, int p_nModelIdx=0);
	virtual void UpdateModels();
    
    ///FOR CUDA
    void EMInitalize();
    ///FOR CUDA
private:
    int numObjs;
    int numDims;
    int numClusters;
    
    float * device_dimObjects;
    float ** dimObjects;
    
    float **dimClusters;
    float * device_dimClusters;
    
    float **dimDiagCovs;
    float * device_diagCovs;
    
    float *dMats;
    float *device_dMats;
    
    float **probabilities;
    float *device_probabilities;
    
    float *dOcc;
    float totalOcc;
    
    float *device_reduceOcc[2];
    
    float *device_meanOcc[2];
    float *meanOcc;
    float *device_varOcc[2];
    float *varOcc;
    
};

#endif // _GMMTRAIN_H_
