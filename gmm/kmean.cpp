#include "kmean.h"

#include "Feat_comm.h"
#include "wtime.h"

#define LOG2PI (log(2*PI))

KMean::KMean() {
    m_WeightIndex = NULL;
    memberShip = NULL;
}

KMean::~KMean()
{
    if(m_WeightIndex) {
        free(m_WeightIndex);
        m_WeightIndex=NULL;	
    }
    if(memberShip) {
        free(memberShip);
        memberShip = NULL;
    }
}

int KMean::SingleChoose(float* sfeat,GaussMixModel * GMMModel,int ModelIndex)
{
    float MinDis=100000000000.0f;
    int   MinIndex;
    for(int m=0;m<GMMModel[ModelIndex].nMixNum;m++)
    {
        float TempDis=0.0f;
        for(int i=0;i<m_nVecSize;i++)
        {
            float t=0.0f;
            t=sfeat[i]-GMMModel[ModelIndex].pGauss[m].pfMean[i];
            TempDis+=t*t;
        }
        if(TempDis<MinDis)
        {
            MinDis=TempDis;
            MinIndex=m;
        }
    }
    return MinIndex;
}


void KMean::KMeanCluster()
{
    int ModelIndex = 0;
    GaussMixModel * tmpGMM;
    tmpGMM=(GaussMixModel*)malloc(sizeof(GaussMixModel));
    AllocGaussMixModel(tmpGMM,m_nMixNum,m_nVecSize);
    for(int m=0;m<m_nMixNum;m++)
    {
        for(int j=0;j<m_nVecSize;j++)
        {
            tmpGMM[ModelIndex].pGauss[m].pfMean[j]=m_pGmmModel[ModelIndex].pGauss[m].pfMean[j];
            m_pGmmModel[ModelIndex].pGauss[m].pfMean[j]=0.0f;
        }

        // 2007.09.17 plu : 
        tmpGMM[ModelIndex].nMixNum = m_pGmmModel[ModelIndex].nMixNum;
    }

    if(m_WeightIndex) {
        free(m_WeightIndex);
        m_WeightIndex = NULL;
    }
    m_WeightIndex=(long *)malloc(sizeof(long)*m_pGmmModel[ModelIndex].nMixNum);
    memset(m_WeightIndex,0,sizeof(long)*m_pGmmModel[ModelIndex].nMixNum);
    long m_TotFrameNum=0;
    
    int fix = 0;
    for(int fileIdx = 0; fileIdx < features->nFeatures; fileIdx++) {
        int m_TempFrameNum = features->featureSize[fileIdx];
        float * feat = features->features[fileIdx];

        for(int f=0;f<m_TempFrameNum;f++)
        {
            float *tmpbuf=feat+f*m_nVecSize4;
            int tmpIndex=SingleChoose(tmpbuf,tmpGMM,ModelIndex);
            memberShip[fix++] = tmpIndex;
            for(int j=0;j<m_nVecSize;j++)
            {
                m_pGmmModel[ModelIndex].pGauss[tmpIndex].pfMean[j]+=tmpbuf[j];
            }
            m_WeightIndex[tmpIndex]++;
        }
        m_TotFrameNum+=m_TempFrameNum;
    }
    FreeGaussMixModel(tmpGMM,m_nModelNum);
    
    for(int m=0;m<m_pGmmModel[ModelIndex].nMixNum;m++)
    {
        m_pGmmModel[ModelIndex].pfWeight[m]=float(m_WeightIndex[m])/float(m_TotFrameNum);
    }
}

void KMean::GetKMeanModel() {
    int ModelIndex = 0;
    for(int m=0;m<m_pGmmModel[ModelIndex].nMixNum;m++)
    {
        for(int j=0;j<m_nVecSize;j++)
        {	
            m_pGmmModel[ModelIndex].pGauss[m].pfMean[j]/=m_WeightIndex[m];
            m_pGmmModel[ModelIndex].pGauss[m].pfDiagCov[j]=0.0f;
        }
    }

    int fix = 0;
    for(int fileIdx = 0; fileIdx < features->nFeatures; fileIdx ++) {
        int m_TempFrameNum = features->featureSize[fileIdx];
        float * feat= features->features[fileIdx]; //(float*)malloc(sizeof(float)*m_TempFrameNum*m_nVecSize4);

        for(int f=0;f<m_TempFrameNum;f++)
        {
            float *tmpbuf=feat+f*m_nVecSize4;
            int tmpIndex=memberShip[fix++]; //SingleChoose(tmpbuf,m_pGmmModel,ModelIndex);
            
            for(int j=0;j<m_nVecSize;j++)
            {
                float t=tmpbuf[j] - m_pGmmModel[ModelIndex].pGauss[tmpIndex].pfMean[j];
                m_pGmmModel[ModelIndex].pGauss[tmpIndex].pfDiagCov[j]+=t*t;
            }
        }
    }
    
    for(int m=0;m<m_pGmmModel[ModelIndex].nMixNum;m++) {
        m_pGmmModel[ModelIndex].pGauss[m].dMat=0.0;
        for(int j=0;j<m_nVecSize;j++)
        {
            m_pGmmModel[ModelIndex].pGauss[m].pfDiagCov[j]/=m_WeightIndex[m];
            m_pGmmModel[ModelIndex].pGauss[m].pfDiagCov[j]=1.0/m_pGmmModel[ModelIndex].pGauss[m].pfDiagCov[j];
            m_pGmmModel[ModelIndex].pGauss[m].dMat+=log(m_pGmmModel[ModelIndex].pGauss[m].pfDiagCov[j]);
        }
        m_pGmmModel[ModelIndex].pGauss[m].dMat-=m_nVecSize*LOG2PI;
        m_pGmmModel[ModelIndex].pGauss[m].dMat*=0.5;
        m_pGmmModel[ModelIndex].pGauss[m].dMat+=log(m_pGmmModel[ModelIndex].pfWeight[m]);
    }
}

void KMean::DataPrepare(struct Features &features, int MaxMixNum) {
    this->features = &features;
    
    numObjs = 0;
    for(int idx = 0;idx < features.nFeatures;idx++) {
        numObjs += features.featureSize[idx];
    }

    memberShip = (int *) malloc(sizeof(int) * numObjs);

}
void KMean::KMeanIteration() {
    KMeanCluster();

    GetKMeanModel();
}
    
void KMean::KMeanMain(int KMeanIterNum) {
    for(int i=0;i < KMeanIterNum;i++) {
        KMeanIteration();
    }
}
