#include "Feat_comm.h"
#include "GMMParam.h"

class KMean : public GMMParam
{
public:
    void KMeanMain(int KMeanIterNum);
    void DataPrepare(struct Features &features, int MaxMixNum);

	KMean();
	~KMean();
	long * m_WeightIndex;
    
protected:

private:
	void KMeanCluster();
	void KMeanCluster2();
	void KMeanCluster3();
    void KMeanIteration();
	int SingleChoose(float* feat,GaussMixModel * GMMModel,int ModelIndex);
	void GetKMeanModel();
	void GetKMeanModel2();
private:
    struct Features *features;

    
    /// For CUDA 
private:
    void KMeanInitalize();
    void KMeanFinalize();
    /// For CUDA 
private:
    int numObjs;
    int numDims;
    float ** dimObjects;
    
    float * device_dimObjects;
    
    int * memberShip;
    int * device_memberShip;
    
    int numClusters;
    float **dimClusters;
    float * device_dimClusters;
    
    float **dimDiagCovs;
    float * device_diagCovs;
    
    int * clusterSize;
    int * device_clusterSize;
    
    float *dMats;
    float *device_dMats;

};
