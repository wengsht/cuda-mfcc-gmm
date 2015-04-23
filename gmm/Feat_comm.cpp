// =====================================================================================
// 
//       Filename:  Feat_comm.cpp
// 
//    Description:  
//    
// 
//        Version:  0.01
//        Created:  04/13/2015 03:35:30 PM
//       Revision:  none
//       Compiler:  clang 3.5
// 
//         Author:  wengsht (SYSU-CMU), wengsht.sysu@gmail.com
//        Company:  
// 
// =====================================================================================

#include "Feat_comm.h"

#include "stdio.h"

#include "comm_srlr.h"

void ReadFeatures(char *p_pcFlistFile, struct Features &features, int &maxFrameNum, int featDim) {
    FILE *fpFeatList;

    TextReadOpen(fpFeatList,p_pcFlistFile);

    int &nFiles = features.nFeatures;

    fscanf(fpFeatList, "%d", &nFiles);

    features.featureDim = featDim;
    
    float **featuresMat = (features.features) = (float **) malloc(sizeof(float *) * nFiles); 
    int * featureSize   = (features.featureSize) = (int *) malloc(sizeof(int) * nFiles);

    int nNum=0,nLen,nTotalFrmNum=0,nFrmNum;
    char Line[512];

    maxFrameNum = 0;
    
    for(int fileIdx = 0; fileIdx < nFiles; fileIdx ++) {
        if (fscanf(fpFeatList,"%s",Line)!=EOF) {
            if (strlen(Line)<=1) continue;		// ¿ÕÐÐ

    //        ReadFeatFile_size1(Line,nFrmNum);
            ReadFeature(Line, &featuresMat[fileIdx], &featureSize[fileIdx]);
            
            maxFrameNum = max(maxFrameNum, featureSize[fileIdx]);

            nTotalFrmNum += nFrmNum;
        }
    }
    fclose(fpFeatList);
}   

void ReadFeature(char *p_pcFeatFile, float **feature, int *featureSize) {
    FILE *fp_feat;
    Feature_BaseInfo p_sFeatInfo;
    ReadOpen(fp_feat,p_pcFeatFile);

    float * p_pfFeatBuf;

    fread(&p_sFeatInfo,sizeof(Feature_BaseInfo),1,fp_feat);
    *featureSize = p_sFeatInfo.nFrameNum;
    
    fseek(fp_feat,FILE_HEADER_SIZE,SEEK_SET);

    p_pfFeatBuf = (*feature) = (float *)malloc(p_sFeatInfo.nTotalParamSize);
    
    int nNumRead = fread(p_pfFeatBuf,1,p_sFeatInfo.nTotalParamSize,fp_feat);
    
    fclose(fp_feat);
}

void FreeFeatures(struct Features &features) {
    free(features.featureSize);
    for(int fileIdx = 0; fileIdx < features.nFeatures; fileIdx ++) {
        free(features.features[fileIdx]);
    }
    free(features.features);
    memset(&features, 0, sizeof(features));
}
