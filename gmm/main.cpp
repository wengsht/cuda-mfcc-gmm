#include <time.h>

#include "Feat_comm.h"
#include "comm_srlr.h"
#include "GMMTrain.h"
#include "GMMMixup.h"
#include "config.h"
#include "kmean.h"
#include "wtime.h"

GMMTrain * pGMMTrain;
GMMMixup * pGMMMixUp;
KMean    * pKMeanGMM; 

void IniGMM(struct Features &features,int VecSize, int MaxMixNum, GMMParam *gmm_params);
int  EMtrain(GMMParam & gmm_params, int ItlerNum) ;  
void MixUpGMM(GMMParam & gmm_params,int NewMixNum,float perDept);  
void KMeanGMM(GMMParam & gmm_params, int KMeanIterNum);

int main(int argc,char * argv[]) {
    double startT = wtime();
    if (argc < 3) {
        printf("Usage: %s <cfgFile> <outputModelFile>\n",argv[0]);
        printf("The config file format:\n");
        printf("UBMRoot=<...>\n");
        printf("MfcList=<...>\n");
        printf("SplitDepth=<...>\n");
        printf("MaxMixNum=<...>\n");
        printf("VecSize=<...>\n");
        printf("MinOccFloorFact=<...>\n");
        printf("WFloor=<...>\n");
        printf("VarFloor=<...>\n");
        printf("EMIlterNum=<...>\n");
        printf("KMeanIterNum=<...>\n");
        printf("LastEMIlterNum=<...>\n");

        return -1;
    }

    char UBMRoot[256];

    int NewMixNum=0;         
    int OldMixNum=0;
    int MaxMixNum=0;
    int EMIlterNum=0;
    int LastEMIlterNum=0;
    int KMeanIterNum=0;

    char MfcList[256];
    char TempMfc[256];
    int VecSize=0,VecSize4;
    float SplitDepth=0.2;
    float fMinOccFact=0.0,fWFloor=0.0,fVarFloor=0.0;

    struct Features features;
    int maxFrameNum;

    GMMParam gmm_params;

    /// Read Configurations 
    Config cfg;
    if(!cfg.SetConfigFile(argv[1])) {
        printf("Error read config : %s\n",argv[1]);
        exit(-1);
    }

    cfg.ReadConfig("UBMRoot",UBMRoot);    /* Output directory */
    cfg.ReadConfig("MfcList",MfcList);    /* Mfcc File list */
    cfg.ReadConfig("SplitDepth",SplitDepth); /* Split Kmean Depth */
    cfg.ReadConfig("MaxMixNum",MaxMixNum);
    cfg.ReadConfig("VecSize",VecSize);
    cfg.ReadConfig("MinOccFloorFact",fMinOccFact);
    cfg.ReadConfig("WFloor",fWFloor);
    cfg.ReadConfig("VarFloor",fVarFloor);
    cfg.ReadConfig("EMIlterNum",EMIlterNum);
    cfg.ReadConfig("KMeanIterNum",KMeanIterNum);
    cfg.ReadConfig("LastEMIlterNum",LastEMIlterNum);

    //*  Read the features into memory */
    ReadFeatures(MfcList, features, maxFrameNum, VecSize);

    VecSize4=ALIGN_4F(VecSize);

    IniGMM(features, VecSize, MaxMixNum, & gmm_params);

    pGMMTrain=new GMMTrain();
    pGMMMixUp=new GMMMixup();
    pKMeanGMM = new KMean();

    pGMMTrain->DataPrepare(features, MaxMixNum);
    pGMMTrain->AllocStaticBuf(1, MaxMixNum, VecSize4, maxFrameNum);
    pGMMTrain->SetFloor(fMinOccFact,fWFloor,fVarFloor);

    pKMeanGMM->DataPrepare(features, MaxMixNum);

    OldMixNum=1;

    while (OldMixNum < MaxMixNum) {
        OldMixNum = EMtrain(gmm_params, EMIlterNum);

        NewMixNum = min(OldMixNum * 2, MaxMixNum);

        MixUpGMM(gmm_params, NewMixNum,SplitDepth);

        KMeanGMM(gmm_params,  KMeanIterNum);

        OldMixNum=NewMixNum;

        printf("*************Split MixNum = %d Finished ******************\n",NewMixNum);
    }

    OldMixNum=EMtrain(gmm_params, LastEMIlterNum);

    printf("all training has been finished\n");
    printf("Last Gmm MixtureNum=%d\n",OldMixNum);
    
    delete pGMMTrain;
    delete pGMMMixUp;
    delete pKMeanGMM;

    FreeFeatures(features);
    
    gmm_params.WriteModel(argv[2]);
    
    gmm_params.FreeAll();
    
    double endT = wtime();

    printf("Whole runtime : %f\n", endT - startT);
}


int EMtrain(GMMParam &gmm_params,int ItlerNum) {
    double avgTime = 0;
    int    iteCnt  = 0;
    
    avgTime = 0;
    iteCnt = 0;

    pGMMTrain->LoadModel(gmm_params);
    

    double startT = wtime();
    pGMMTrain->EMTrain(ItlerNum);
    double endT = wtime();
    
    //for(iEMIndex=1;iEMIndex<=ItlerNum;iEMIndex++) {
     //   pGMMTrain->EMIteration();
        
    avgTime += endT - startT;
    iteCnt ++;
    
   // }
    printf("Avg EM time %lf\n", avgTime / iteCnt);
    gmm_params.LoadModel(*pGMMTrain);

    return pGMMTrain->GetMixtureNum();
}

void MixUpGMM(GMMParam & gmm_params,int NewMixNum,float perDept) {
    pGMMMixUp->LoadModel(gmm_params);

    double startT = wtime();
    pGMMMixUp->Mixup(NewMixNum,perDept);
    double endT = wtime();
        
    printf("Avg MixUp time %d, %lf\n", NewMixNum, endT-startT);

    gmm_params.LoadModel(*pGMMMixUp);
}

void KMeanGMM(GMMParam & gmm_params, int KMeanIterNum) {
    pKMeanGMM->LoadModel(gmm_params);

    double avgTime = 0;
    int    iteCnt  = 0;
    
    
    double startT = wtime();
    pKMeanGMM->KMeanMain(KMeanIterNum);
    double endT = wtime();

    avgTime += endT - startT;
    iteCnt ++;
    
    printf("Avg KMean time %lf\n", avgTime / iteCnt);
    printf("Avg KMean Iteration: %f\n", avgTime / iteCnt / KMeanIterNum);
    gmm_params.LoadModel(gmm_params);
}

void IniGMM(struct Features &features,int VecSize, int MaxMixNum, GMMParam * gmm_params) {
    gmm_params->AllocAll(MaxMixNum, VecSize);

    gmm_params->InitModel(features, VecSize, ALIGN_4F(VecSize));
}
