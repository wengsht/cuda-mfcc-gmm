//
//  Feature.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "Feature.h"
#include "tool.h"

#include <cmath>

const double Feature::IllegalDist = 1e18;
double Feature::operator -(const Feature & T) {
    double res = 0.0;

    if(size() != T.size()) {
        Warn("Operator - on Features should have same size\n");

        return Feature::IllegalDist;
    }

    int idx, siz = size();
    for(idx = 0; idx < siz; idx ++) {
        res += pow(data[idx] - T[idx], 2.0);
    }
    res = sqrt(res);

    return res;
}
void Feature::push_back(double d) {
    data.push_back(d);
}

Feature::FeatureType Feature::getFeatureType() {
    return type;
}
Feature::FeatureType Feature::setFeatureType(Feature::FeatureType type) {
    this->type = type;

    return type;
}

SP_RESULT Feature::fillDelta() {
    if(Raw != type) {
        return SP_FEATURE_NOT_RAW;
    }

    int siz = data.size();
    data.resize(2 * siz);

    dumpDelta(0, siz, siz);

    type = Delta;

    return SP_SUCCESS;
}
SP_RESULT Feature::fillDoubleDelta() {
    if(Raw != type) {
        return SP_FEATURE_NOT_RAW;
    }

    int siz = data.size();
    data.resize(3 * siz);

    dumpDelta(0, siz, siz);
    dumpDelta(siz, 2*siz, siz);

    type = DoubleDelta;

    return SP_SUCCESS;
}
