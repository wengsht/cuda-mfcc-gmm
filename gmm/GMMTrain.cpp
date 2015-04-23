#include "GMMTrain.h"

#include <cstdlib>
#include <math.h>
#include "assert.h"

#include "Feat_comm.h"
#include "wtime.h"

#define e 2.718281828459

GMMTrain::GMMTrain() {	
    m_pGMMStatic = NULL;

    m_nMaxFrameNum = 0;

    //m_pfFeatBuf = NULL;

    m_fWeightFloor = -1.f;		// 表示无效
    m_pfVarFloor = NULL;
    m_fOccFloor = -1.f;

    m_ppfProb_BufBuf = NULL;	// 两个中间buf
    m_pfProb_Buf = NULL;

    // 2007.09.16 plu : 
    m_pfIPPBuf_VecSize4 = NULL;
}

void GMMTrain::DataPrepare(struct Features &features ,int MaxMixNum) {
    this->features = &features;
}

GMMTrain::~GMMTrain()
{
    FreeStaticBuf();

    //	if (NULL!=m_pfFeatBuf)	Free(m_pfFeatBuf);
    if (NULL!=m_pfVarFloor) Free(m_pfVarFloor);

    if (NULL!=m_pfProb_Buf) 
        Free(m_pfProb_Buf);

    if (NULL!=m_ppfProb_BufBuf) 
        Free((void **)m_ppfProb_BufBuf,m_nMixNum);

    //m_pfFeatBuf=NULL;
    m_pfVarFloor=NULL;
    m_pfProb_Buf=NULL;
    m_ppfProb_BufBuf=NULL;
}


// 2007.03.16 plu : 设置门限
void GMMTrain::SetFloor(float p_fOccFloor,float p_fWeightFloor,float p_fVarFloor)
{
    ASSERT2(p_fVarFloor>0.f,"Error call SetFloor() : p_fVarFloor<0!");
    ASSERT2(p_fOccFloor>0.f,"Error call SetFloor() : p_fOccFloor<0!");
    ASSERT2(p_fWeightFloor>0.f,"Error call SetFloor() : p_fWeightFloor<0!");

    m_fOccFloor   = p_fOccFloor;
    m_fWeightFloor= p_fWeightFloor;
    m_fVarFloor   = p_fVarFloor;
}

// 分配存储中间量的ProbBuf
void GMMTrain::AllocProbBuf(int m_nMaxFrameNum, int m_nMixNum)
{
    ASSERT2(m_nMaxFrameNum>0,"Error call AllocProbBuf() : m_nMaxFrameNum<=0!");
    ASSERT2(m_nMixNum>0,"Error call AllocProbBuf() : m_nMixNum<=0!");

    if(NULL!=m_pfProb_Buf)		Free(m_pfProb_Buf);
    m_pfProb_Buf = (float *)Malloc(sizeof(float)*m_nMaxFrameNum);

    if (NULL!=m_ppfProb_BufBuf)		Free((void **)m_ppfProb_BufBuf,m_nMixNum);
    //	m_ppfProb_BufBuf = (float **)Malloc(m_nMixNum,m_nMaxFrameNum*sizeof(float),true);
    m_ppfProb_BufBuf = (float **)Malloc(m_nMixNum,m_nMaxFrameNum,int(sizeof(float)),true);

}

void GMMTrain::AllocStaticBuf(int m_nModelNum, int m_nMixNum, int m_nVecSize4, int m_nMaxFrameNum)
{
    //	ASSERT2(m_nModelNum>0,"Error call AllocStaticBuf() : m_nModelNum<0!");
    //	ASSERT2(m_nMixNum>0,"Error call AllocStaticBuf() : m_nMixNum<0!");
    //	ASSERT2(m_nVecSize4>0,"Error call AllocStaticBuf() : m_nVecSize4<0!");

    //m_pfFeatBuf = (float *)Malloc(m_nMaxFrameNum*m_nVecSize4*sizeof(float),true);
    AllocProbBuf(m_nMaxFrameNum, m_nMixNum);

    if (NULL!=m_pGMMStatic)
        FreeStaticBuf();

    m_pGMMStatic=(GMMStatic *)Malloc(m_nModelNum*sizeof(GMMStatic));

    for(int i=0;i<m_nModelNum;i++)
    {
        m_pGMMStatic[i].pGauss = (GaussStatis *)Malloc(m_nMixNum*sizeof(GaussStatis));
        for(int m=0;m<m_nMixNum;m++)
        {
            m_pGMMStatic[i].pGauss[m].pfMeanAcc = (float *)Malloc(m_nVecSize4*sizeof(float),true);
            m_pGMMStatic[i].pGauss[m].pfVarAcc = (float *)Malloc(m_nVecSize4*sizeof(float),true);
            m_pGMMStatic[i].pGauss[m].dOcc = 0.0;
        }
        m_pGMMStatic[i].dTotalOcc = 0.0;
        m_pGMMStatic[i].fTotalProb = 0.f;
        m_pGMMStatic[i].nTotalNum = 0;
    }

    // 2007.09.16 plu : 
    m_pfIPPBuf_VecSize4 = (float *)Malloc(sizeof(float)*m_nVecSize4);
}

// 释放统计量buf
void GMMTrain::FreeStaticBuf()
{
    if (NULL!=m_pGMMStatic)
    {
        ASSERT2(m_nModelNum>0,"Error call FreeStaticBuf() : m_nModelNum<=0!");

        for (int i=0;i<m_nModelNum;i++)
        {
            if (NULL!=m_pGMMStatic[i].pGauss)
            {
                for(int m=0;m<m_nMixNum;m++)
                {
                    Free(m_pGMMStatic[i].pGauss[m].pfMeanAcc);
                    Free(m_pGMMStatic[i].pGauss[m].pfVarAcc);
                }

                Free(m_pGMMStatic[i].pGauss);
                m_pGMMStatic[i].pGauss = NULL;
            }
        }

        Free(m_pGMMStatic);
        m_pGMMStatic=NULL;
    }

    // 2007.09.16 plu : 
    if (NULL!=m_pfIPPBuf_VecSize4)	Free(m_pfIPPBuf_VecSize4);
    m_pfIPPBuf_VecSize4 = NULL;
}

// 重置统计量buf
void GMMTrain::ResetStaticBuf()
{
    ASSERT2(m_nModelNum>0,"Error call ResetStaticBuf() : m_nModelNum<0!");
    ASSERT2(m_nMixNum>0,"Error call ResetStaticBuf() : m_nMixNum<0!");
    ASSERT2(m_nVecSize4>0,"Error call ResetStaticBuf() : m_nVecSize4<0!");

    for(int i=0;i<m_nModelNum;i++)
    {
        m_pGMMStatic[i].fTotalProb=0.f;
        m_pGMMStatic[i].dTotalOcc=0.0;
        m_pGMMStatic[i].nTotalNum=0;
        for(int m=0;m<m_nMixNum;m++)
        {
            memset(m_pGMMStatic[i].pGauss[m].pfMeanAcc,0,m_nVecSize4*sizeof(float));
            memset(m_pGMMStatic[i].pGauss[m].pfVarAcc,0,m_nVecSize4*sizeof(float));
            m_pGMMStatic[i].pGauss[m].dOcc = 0.0;
        }
    }
}

void observateProbability(float *m_pfFeats, int step, float *mean, float *var, int vecSize, float *res, int nFrameNum, float weight) {
    float *m_pfFeat;

    for(int i = 0;i < nFrameNum;i++) {
        double pdf = 0.0;
        m_pfFeat = m_pfFeats + vecSize * i;

        for(int j = 0;j < vecSize;j++) {
            pdf -=  0.5 * (m_pfFeat[j] - mean[j]) * (m_pfFeat[j] - mean[j]) * var[j];
        }

        res[i] = pdf + weight;
    }
}

void GMMTrain::ComputeStatiscs_MT(float *m_pfFeatBuf, int nFrameNum, int p_nModelIdx) {
    ASSERT3(p_nModelIdx>=0&&p_nModelIdx<m_nModelNum,
            "Error call ComputeStatiscs() : p_nModelIdx=%d!",p_nModelIdx);
    ASSERT2(m_pGMMStatic,"Error call ComputeStatiscs() : m_pGMMStatic==NULL !");
    ASSERT2(m_pGmmModel,"Error call ComputeStatiscs() : m_pGmmModel==NULL !");

    m_pGMMStatic[p_nModelIdx].nTotalNum += nFrameNum;

    observateProbability(m_pfFeatBuf,
            m_nVecSize4,
            m_pGmmModel[p_nModelIdx].pGauss[0].pfMean,
            m_pGmmModel[p_nModelIdx].pGauss[0].pfDiagCov,
            m_nVecSize,
            m_pfProb_Buf,
            nFrameNum,
            m_pGmmModel[p_nModelIdx].pGauss[0].dMat);

    memcpy(m_ppfProb_BufBuf[0],m_pfProb_Buf,sizeof(float)*nFrameNum);

    for(int m=1;m<m_nMixNum;m++)
    {
        observateProbability(m_pfFeatBuf,
                m_nVecSize4,
                m_pGmmModel[p_nModelIdx].pGauss[m].pfMean,
                m_pGmmModel[p_nModelIdx].pGauss[m].pfDiagCov,
                m_nVecSize,				
                (float *) m_ppfProb_BufBuf[m],
                nFrameNum,
                m_pGmmModel[p_nModelIdx].pGauss[m].dMat);

        for(int j = 0;j < nFrameNum; j++) {
            float a = m_pfProb_Buf[j], b = m_ppfProb_BufBuf[m][j];
            if(a > b) std::swap(a, b);
            m_pfProb_Buf[j] = b + log(1.0 + pow(e, a-b));
        }
    }

    float *pCurFrame=m_pfFeatBuf;
    float fProbSum = 0.f;

    for(int j = 0;j < nFrameNum;j++) {
        fProbSum += m_pfProb_Buf[j];
    }

    m_pGMMStatic[p_nModelIdx].fTotalProb += fProbSum;

    for(int m=0;m<m_nMixNum;m++) {
        fProbSum = 0.0;
        for(int j = 0; j < nFrameNum; j++) {
            m_ppfProb_BufBuf[m][j] -= m_pfProb_Buf[j];

            m_ppfProb_BufBuf[m][j] = pow(e, m_ppfProb_BufBuf[m][j]);

            fProbSum += m_ppfProb_BufBuf[m][j];
        }
        
        m_pGMMStatic[p_nModelIdx].pGauss[m].dOcc += fProbSum;
        m_pGMMStatic[p_nModelIdx].dTotalOcc += fProbSum;

        pCurFrame=m_pfFeatBuf;
        for (int i=0;i<nFrameNum;i++)
        {
            for(int j = 0; j < m_nVecSize4; j++) {
                m_pfIPPBuf_VecSize4[j] = m_ppfProb_BufBuf[m][i] * pCurFrame[j];

                m_pGMMStatic[p_nModelIdx].pGauss[m].pfMeanAcc[j] += m_pfIPPBuf_VecSize4[j];
            }

            for(int j = 0; j < m_nVecSize4; j++) {
                m_pfIPPBuf_VecSize4[j] = pCurFrame[j] * pCurFrame[j] * m_ppfProb_BufBuf[m][i];

                m_pGMMStatic[p_nModelIdx].pGauss[m].pfVarAcc[j] += m_pfIPPBuf_VecSize4[j];
            }

            pCurFrame += m_nVecSize4;
        }
    }
}


// 更新模型
void GMMTrain::UpdateModels()
{
    ASSERT2(m_nModelNum>0,"Error call UpdateModels() : m_nModelNum<0!");
    ASSERT2(m_nMixNum>0,"Error call UpdateModels() : m_nMixNum<0!");
    ASSERT2(m_pGMMStatic,"Error call UpdateModels() : m_pGMMStatic==NULL !");
    ASSERT2(m_pGmmModel,"Error call UpdateModels() : m_pGmmModel==NULL !");

//    printf("wengshtc %f\n", m_pGMMStatic[0].pGauss[0].dOcc);
   // printf("wengshtcocc 0 %f\n", m_pGMMStatic[0].pGauss[0].dOcc);
    //printf("wengshtcmean %f\n", m_pGMMStatic[0].pGauss[0].pfMeanAcc[0]);
    //printf("wengshtcvar %f\n", m_pGMMStatic[0].pGauss[0].pfVarAcc[0]);
    int	  vfloorNum,wfloorNum,nNoUpdateNum;
    float fminOcc,fmaxOcc;

    float		fWeightSum;	

    for(int i=0;i<m_nModelNum;i++)
    {
        fWeightSum=0.0;

        vfloorNum=wfloorNum=nNoUpdateNum=0;

        fminOcc = fmaxOcc = m_pGMMStatic[i].pGauss[0].dOcc;	

        // end if ( NULL!=m_pfVarFloor )
        //printf("NULL==m_pfVarFloor \n");
        for(int m=0;m<m_nMixNum;m++)
        {
            if(m_pGMMStatic[i].pGauss[m].dOcc < fminOcc)
                fminOcc=m_pGMMStatic[i].pGauss[m].dOcc;

            if(m_pGMMStatic[i].pGauss[m].dOcc > fmaxOcc)	
                fmaxOcc=m_pGMMStatic[i].pGauss[m].dOcc;

            if(m_pGMMStatic[i].pGauss[m].dOcc>m_fOccFloor) 
            {
                // 更新weight
                m_pGmmModel[i].pfWeight[m] = m_pGMMStatic[i].pGauss[m].dOcc/m_pGMMStatic[i].dTotalOcc;

                if(m_pGmmModel[i].pfWeight[m]<m_fWeightFloor)
                {
                    m_pGmmModel[i].pfWeight[m]=m_fWeightFloor;
                    wfloorNum++;
                }

                fWeightSum += m_pGmmModel[i].pfWeight[m];

                m_pGmmModel[i].pGauss[m].dMat=0.0;

                double occFact=1.0f/m_pGMMStatic[i].pGauss[m].dOcc;
                for(int j=0;j<m_nVecSize;j++)
                {					
                    // 更新均值
                    m_pGmmModel[i].pGauss[m].pfMean[j] = m_pGMMStatic[i].pGauss[m].pfMeanAcc[j]*occFact;

                    // 更新方差
                    double dVar = m_pGMMStatic[i].pGauss[m].pfVarAcc[j]*occFact
                        - m_pGmmModel[i].pGauss[m].pfMean[j]*m_pGmmModel[i].pGauss[m].pfMean[j];
                    if(dVar<m_fVarFloor)
                    {
                        m_pGmmModel[i].pGauss[m].pfDiagCov[j] = 1.f/m_fVarFloor;
                        vfloorNum++;
                    }
                    else
                        m_pGmmModel[i].pGauss[m].pfDiagCov[j] = 1.f/dVar;

                    m_pGmmModel[i].pGauss[m].dMat += log(m_pGmmModel[i].pGauss[m].pfDiagCov[j]);
                }

                m_pGmmModel[i].pGauss[m].dMat -= m_nVecSize*log2pi;
                m_pGmmModel[i].pGauss[m].dMat *= 0.5;	
            }
            else
            {
                m_pGmmModel[i].pGauss[m].dMat -= log(m_pGmmModel[i].pfWeight[m]); // 从mat中减去log(weight)

///                printf("%d-Model %d-th Gauss no updated : occ=%f\n",i,m,m_pGMMStatic[i].pGauss[m].dOcc);

                nNoUpdateNum++;
                fWeightSum += m_pGmmModel[i].pfWeight[m];
            }// end else

        }// end for(int m=0;m<m_nMixNum;m++)


        float ftmp=0.f;
        for(int m=0;m<m_nMixNum;m++)
        {
            m_pGmmModel[i].pfWeight[m] /= fWeightSum;  // 保证Weight之和等于1

            m_pGmmModel[i].pGauss[m].dMat += log(m_pGmmModel[i].pfWeight[m]);

            if (m_pGmmModel[i].pGauss[m].dMat>=0.0)
                printf("Warning : m_pGmmModel[%d].pGauss[%d].dMat=%.3f!\n",i,m,m_pGmmModel[i].pGauss[m].dMat);
        }

   //     printf("============== update %d-th GMM-Model==========\n",i);
   //     printf("TotalProb=%12.6f, TotalFmNum=%5d, avgProb=%12.6f\n",
   //             m_pGMMStatic[i].fTotalProb,
   //             m_pGMMStatic[i].nTotalNum,
   //             m_pGMMStatic[i].fTotalProb/(float)m_pGMMStatic[i].nTotalNum);

   //     printf("number of no update is %d\n",nNoUpdateNum);
   //     printf("%d floored weight, %d floored variance\n",wfloorNum,vfloorNum);
   //     printf("min mixture occ %f\nmax mixture occ %f\n",fminOcc,fmaxOcc);

    }// end for (int i=0;i<m_nModel;i++)
}

void GMMTrain::EMIteration() {
    ResetStaticBuf();

    //printf("wengshtinit %f %f %f\n", m_pGmmModel[0].pGauss[0].dMat, m_pGmmModel[0].pGauss[0].pfMean[34], m_pGmmModel[0].pGauss[1].pfDiagCov[35]);
    ComputeStatistics();

    UpdateModels();
}


void GMMTrain::ComputeStatistics() {
    for(int fileIdx = 0; fileIdx < features->nFeatures; fileIdx ++) {
        ComputeStatiscs_MT(features->features[fileIdx], features->featureSize[fileIdx]);
    }
}

void GMMTrain::EMTrain(int EMIterNum) {
#ifdef PERFORMANCE_REPORT
    float avg = 0;
    int   cnt = 0;
    double startT = 0, endT = 0;
#endif

    for(int iEMIndex=1 ;iEMIndex<=EMIterNum; iEMIndex++) {
#ifdef PERFORMANCE_REPORT
        startT = wtime();
#endif
        EMIteration();
#ifdef PERFORMANCE_REPORT
        endT   = wtime();
        avg += endT - startT;
        cnt ++;
#endif
    }
#ifdef PERFORMANCE_REPORT
    printf("Avg EM Iteration: %f\n", avg / cnt);
#endif
}
