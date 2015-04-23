/******************************DOCUMENT*COMMENT***********************************
*D
*D 文件名称            : GMMParam.cpp
*D
*D 项目名称            : 
*D
*D 版本号              : 1.1.0003
*D
*D 文件描述            :
*D
*D
*D 文件修改记录
*D ------------------------------------------------------------------------------ 
*D 版本号       修改日期       修改人     改动内容
*D ------------------------------------------------------------------------------ 
*D 1.1.0001     2007.01.20     plu        创建文件
*D 1.1.0002     2007.09.11     plu        增加MFCC类型码检查，及将该类型码写入GMM param文件
*D 1.1.0003     2007.09.17     plu        改变GMM模型内存分配方式，所有的均值放在一块，所有的方差放在一块
*D*******************************************************************************/
#include "GMMParam.h"

#include "comm_srlr.h"

GMMParam::GMMParam()
{
	m_nVecSize4 = m_nVecSize = -1;
	m_nModelNum = m_nMixNum = -1;
		
	m_pGmmModel = NULL;
	m_fpLog = NULL;   // 2007.03.20 plu : 

	m_nMfccKind = -1;			// 2007.09.11 plu : 
}

GMMParam::~GMMParam()
{   
	if (NULL!=m_fpLog) fclose(m_fpLog);   // 2007.03.20 plu : 
   //Y if(m_pGmmModel) {
   //Y     FreeGaussMixModel(m_pGmmModel,m_nModelNum);
   //Y }
}

// 释放模型
void GMMParam::FreeGaussMixModel(GaussMixModel *p_gmmParam,int p_nModelNum)
{
	if (p_gmmParam)
	{
		ASSERT2(p_nModelNum,"Error call FreeGaussMixModel() : p_nModelNum<0!");
		for (int i=0;i<p_nModelNum;i++)
		{
			if (p_gmmParam[i].pGauss)		
			{
				/*// plu 2007.09.17_16:08:11
				for (int m=0;m<p_gmmParam[i].nMixNum;m++)
				{
					Free(p_gmmParam[i].pGauss[m].pfMean);
					Free(p_gmmParam[i].pGauss[m].pfDiagCov);
				}
				*/// plu 2007.09.17_16:08:11

				// 2007.09.17 plu : 
				Free(p_gmmParam[i].pfMeanBuf);
				Free(p_gmmParam[i].pfDiagCovBuf);

				Free(p_gmmParam[i].pGauss);

				p_gmmParam[i].pfMeanBuf = NULL;
				p_gmmParam[i].pfDiagCovBuf = NULL;
				p_gmmParam[i].pGauss = NULL;
			}
			Free(p_gmmParam[i].pfWeight);
			p_gmmParam[i].pfWeight = NULL;
		}
		Free(p_gmmParam);
		p_gmmParam = NULL;
	}
}

// 分配一个GMM模型的空间
void GMMParam::AllocGaussMixModel(GaussMixModel *p_pModel,int p_nMixNum,int p_nVecSize)
{
	ASSERT2(p_nMixNum,"Error call AllocGaussMixModel() : p_nMixNum<0!");
	ASSERT2(p_nVecSize,"Error call AllocGaussMixModel() : p_nVecSize<0!");
	
	// 分配
	p_pModel->pfWeight = (float *)malloc(p_nMixNum*sizeof(float));
	p_pModel->pGauss = (GaussPDF*)malloc (p_nMixNum*sizeof(GaussPDF));

	/*// plu 2007.09.17_16:04:42
	for(int m=0;m<p_nMixNum;m++)
	{
		p_pModel->pGauss[m].pfMean=(float *)Malloc (p_nVecSize*sizeof(float),true);
		p_pModel->pGauss[m].pfDiagCov=(float *)Malloc (p_nVecSize*sizeof(float),true);
	}
	*/// plu 2007.09.17_16:04:42

	// 2007.09.17 plu : 连续分配均值、方差的内存空间
	p_pModel->pfMeanBuf = (float *)Malloc(p_nMixNum*sizeof(float)*p_nVecSize,true);
	p_pModel->pfDiagCovBuf = (float *)Malloc(p_nMixNum*sizeof(float)*p_nVecSize,true);
	for(int m=0;m<p_nMixNum;m++)
	{
		// 对于每个高斯分量的均值和方差，赋内存的首地址
		p_pModel->pGauss[m].pfMean = p_pModel->pfMeanBuf + m*p_nVecSize;
		p_pModel->pGauss[m].pfDiagCov = p_pModel->pfDiagCovBuf + m*p_nVecSize;
	}

	// 初始化
	p_pModel->pfWeight[0] = 1.f;
}


// 加载GMM模型
// 注意: 目前的版本是将m_nVecSize4维的矢量写入model文件！！！
void GMMParam::LoadModel(char *p_pcModelFile)
{
	ASSERT2(p_pcModelFile,"Error call LoadModel() : p_pcModelFile=NULL!");

	// 打开模型文件
	FILE *fpModel;
	ReadOpen(fpModel,p_pcModelFile);

	// 读出并检查文件头
	GMMFileHeader  ModelHeader;					// Header *header; 模型文件的文件头
	fread(&ModelHeader,sizeof(GMMFileHeader),1,fpModel);
	ASSERT3(ModelHeader.nDim>0, "Error in model file %s : nDim<0!",p_pcModelFile);
	ASSERT3(ModelHeader.nMixNum>0, "Error in model file %s : nMixNum<0!",p_pcModelFile);
	ASSERT3(ModelHeader.nModelNum>0, "Error in model file %s : nModelNum<0!",p_pcModelFile);

	// 设定 矢量维数，混合高斯数目，模型数目
	m_nVecSize = ModelHeader.nDim;
	m_nVecSize4 = ALIGN_4F(m_nVecSize);
	m_nMixNum  = ModelHeader.nMixNum;		// 文件p_pcModelFile中的每个模型的mix数目现在都是一样的，合理？
	m_nModelNum  = ModelHeader.nModelNum;

	// 分配内存
//Y	if (NULL != m_pGmmModel)
//Y		FreeGaussMixModel(m_pGmmModel,m_nModelNum);
	m_pGmmModel = (GaussMixModel *) Malloc(m_nModelNum,sizeof(GaussMixModel),false);

	for(int i=0;i<m_nModelNum;i++)
	{
		// 分配 GMM 模型
		AllocGaussMixModel(&m_pGmmModel[i],m_nMixNum,m_nVecSize4);

		m_pGmmModel[i].nMixNum = m_nMixNum;

		// 读出weight
		fread(m_pGmmModel[i].pfWeight,sizeof(float),m_nMixNum,fpModel);

		// 读出均值，斜方差，mat
		for(int m=0;m<m_nMixNum;m++)
		{
			m_pGmmModel[i].pGauss[m].nDim = m_nVecSize;

			fread(m_pGmmModel[i].pGauss[m].pfMean,sizeof(float),m_nVecSize4,fpModel);
			fread(m_pGmmModel[i].pGauss[m].pfDiagCov,sizeof(float),m_nVecSize4,fpModel);
			fread(&m_pGmmModel[i].pGauss[m].dMat,sizeof(double),1,fpModel);
		
			// 如果mat值大于零，可能模型有问题，报警
			if (m_pGmmModel[i].pGauss[m].dMat>=0.0)
				printf("Warning : m_pGmmModel[%d].pGauss[%d].dMat=%.3f!\n",i,m,m_pGmmModel[i].pGauss[m].dMat);
		}
	}
	fclose(fpModel);
}


// 注意: 目前的版本是将m_nVecSize4维的矢量写入model文件！！！
bool GMMParam::WriteModel(char *p_pcModelFile)
{
	if (m_nVecSize<=0)
	{
		printf("Error call WriteModel() : m_nVexSize<=0!");
		return false;
	}
	if (m_nMixNum<=0)
	{
		printf("Error call WriteModel() : m_nMixNum<=0!");
		return false;
	}
	if (m_nModelNum<=0)
	{
		printf("Error call WriteModel() : m_nModelNum<=0!");
		return false;
	}
	if (NULL==m_pGmmModel)
	{
		printf("Error call WriteModel() : m_pGmmModel=NULL!");
		return false;
	}

	FILE *fpModel;

	// 打开文件
	WriteOpen(fpModel,p_pcModelFile);

    m_nModelNum = 1;
	// 写文件头
	fwrite(&m_nModelNum,1,sizeof(int),fpModel);
	fwrite(&m_nMixNum,1,sizeof(int),fpModel);
	fwrite(&m_nVecSize,1,sizeof(int),fpModel);
	
	int nOrthogonal;
	/*// plu 2007.09.11_17:07:44
	fwrite(&nOrthogonal,1,sizeof(int),fpModel);		// 为了和以前的模型一致，现在这个变量无用
	*/// plu 2007.09.11_17:07:44
	nOrthogonal = (int)m_nMfccKind;
	fwrite(&nOrthogonal,1,sizeof(int),fpModel);		// 将特征类型写入文件

	for(int i=0;i<m_nModelNum;i++)
	{
		// 写weight
		fwrite(m_pGmmModel[i].pfWeight,sizeof(float),m_nMixNum,fpModel);
		
		for(int m=0;m<m_nMixNum;m++)
		{
			fwrite(m_pGmmModel[i].pGauss[m].pfMean,sizeof(float),m_nVecSize4,fpModel);
			fwrite(m_pGmmModel[i].pGauss[m].pfDiagCov,sizeof(float),m_nVecSize4,fpModel);
			fwrite(&(m_pGmmModel[i].pGauss[m].dMat),sizeof(double),1,fpModel);
		}
	}
	
	fclose(fpModel);

	return true; 
} 

// 2007.03.20 plu : add
void GMMParam::SetLogFile(char *p_pcLogFile)
{
	if (NULL!=m_fpLog)	fclose(m_fpLog);

	// log文件
	m_fpLog=fopen(p_pcLogFile,"wt");
	ASSERT3(m_fpLog,"Error open %s for write!",p_pcLogFile)
}

void GMMParam::AllocAll(int p_nMixNum, int p_nVecSize) {
    if(m_pGmmModel) {
        FreeGaussMixModel( m_pGmmModel, 1 );
    }
    
    m_pGmmModel = (GaussMixModel *) malloc(sizeof(GaussMixModel));
    
    AllocGaussMixModel(m_pGmmModel, p_nMixNum, p_nVecSize);
}

void GMMParam::FreeAll() {
    if(m_pGmmModel) {
        FreeGaussMixModel( m_pGmmModel, 1 );
    }
}

void GMMParam::InitModel(struct Features &features, int vecSize, int vecSize4) {
    m_nModelNum = 1; //modelNum;
    m_nMixNum   = 1; //mixNum;
    m_nVecSize  = vecSize;
    m_nVecSize4    = vecSize4;


    m_pGmmModel[0].pfWeight[0] = 1.0f; 
    

    m_pGmmModel[0].nMixNum = 1;
    float *mean = m_pGmmModel[0].pfMeanBuf;
    float *diagCov = m_pGmmModel[0].pfDiagCovBuf;
    
    float * feature;
    float val;
    int featureSize;
    int featDim = features.featureDim;
    int wholeCnt = 0;
    
    for(int fileIdx = 0; fileIdx < features.nFeatures; fileIdx ++) {
        feature = features.features[fileIdx];
        
        featureSize = features.featureSize[fileIdx];
        
        wholeCnt += featureSize;
        
        for(int featIdx = 0; featIdx < featureSize; featIdx ++) {
            for(int featDimIdx = 0; featDimIdx < featDim; featDimIdx ++) {
                val = feature[featIdx * featDim + featDimIdx]; 
               mean[featDimIdx] += val;
               diagCov[featDimIdx] += val * val;
            }
        }
    }
    for(int featDimIdx = 0; featDimIdx < featDim; featDimIdx ++) {
        mean[featDimIdx] /= wholeCnt;
        diagCov[featDimIdx] /= wholeCnt;

        diagCov[featDimIdx] -= mean[featDimIdx] * mean[featDimIdx];
        diagCov[featDimIdx] = 1.f / diagCov[featDimIdx];
    }
    
    float &dMat = m_pGmmModel[0].pGauss[0].dMat;
    dMat = 0;
    
    for (int i=0;i<featDim;i++)
        dMat += (diagCov[i]<=0.f)?LZERO:log(diagCov[i]);
    dMat -= featDim*log(TPI);
    dMat *= 0.5;
}

void GMMParam::LoadModel(const GMMParam &gmmParam) {
    m_nModelNum = gmmParam.m_nModelNum;
    m_nMixNum = gmmParam.m_nMixNum;
    m_nVecSize = gmmParam.m_nVecSize;
    m_nVecSize4 = gmmParam.m_nVecSize4;
    
    m_pGmmModel = gmmParam.m_pGmmModel;
}
void GMMParam::WriteModel(const GMMParam &gmmParam) {
//    gmmParam.LoadModel(*this);
}

void GMMParam::report() {
    printf("Mixture Number: %d\n", m_nMixNum);
    printf("Feature Width: %d\n", m_nVecSize);
    
    for(int i = 0; i < m_nMixNum; i++) {
        printf("%dth Gaussian Model\n", i);

        printf("MEAN: \n");
        for(int j = 0;j < m_nVecSize; j++) {
            printf("%f ", m_pGmmModel[0].pGauss[i].pfMean[j]);
        }
        puts("");
        
        printf("Diag Var: \n");
        for(int j = 0;j < m_nVecSize; j++) {
            printf("%f ", m_pGmmModel[0].pGauss[i].pfDiagCov[j]);
        }
        puts("");
    }
}
