//
//  Feature.h
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#ifndef __SpeechRecongnitionSystem__Feature__
#define __SpeechRecongnitionSystem__Feature__

#include <vector>
#include <iostream>
#include "tool.h"
// feature is a vector
class Feature{
protected:
    std::vector<double> data;


public:
    // a < b
    static bool better(double a, double b) {
        if(b == Feature::IllegalDist || ( a != Feature::IllegalDist && a < b))
            return true;

        return false;
    }
    const static double IllegalDist;
    enum FeatureType {
        Raw,
        Delta,
        DoubleDelta
    };

    Feature() {
        type = Raw;
    }

    Feature operator * (const Feature & T)const{
        Feature ret;
        
        return ret;
    }
    
    Feature sqr(){
        Feature ret;
        return ret;
    }
    
    Feature operator + (const Feature & T)const{
        Feature ret;
        
        return ret;
    }

    // 重载， 计算两个向量的距离, 取负实现越大越好
    double operator - (const Feature & T);

    FeatureType getFeatureType();
    FeatureType setFeatureType(FeatureType type);

    SP_RESULT fillDelta();
    SP_RESULT fillDoubleDelta();

    void push_back(double d);
    
    Feature operator * (double c)const{
        Feature ret;
        
        return ret;
    }
    void resize(int s) {
        data.resize(s);
    }
    const double &operator[] (int inx) const {
        return data[inx];
    }
    double &operator[] (int inx) {
        return data[inx];
    }
    double *rawData() {
        return data.data();
    }
    int size() const {
        return data.size();
    }

private:
    void dumpDelta(int from, int to, int siz) {
        if(siz <= 1) 
            return ;

        int idx;
        data[to] = data[from + 1];
        data[to+siz-1] = - data[from+siz-2];

        for(idx = 1, to++, from++, from++; idx < siz-1;idx ++, to++, from ++) {
            data[to] = data[from] - data[from-2];
        }
    }
    FeatureType type;
};
#endif /* defined(__SpeechRecongnitionSystem__Feature__) */
