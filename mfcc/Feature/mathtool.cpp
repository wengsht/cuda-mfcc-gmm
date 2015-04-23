//
//  mathtool.cpp
//  SpeechRecongnitionSystem
//
//  Created by Admin on 9/11/14.
//  Copyright (c) 2014 Admin. All rights reserved.
//

#include "mathtool.h"

#include <assert.h>

#include <iostream>
#include "math.h"
#include "Feature.h"
using namespace std;
const int MAXN = 1000;
// fft(a, n, 1) -- dft
// fft(a, n, -1) -- idft
// n should be 2^k
void fft(cp *a,int n,int f)
{
 //   assert(MAXN > n);
    cp *b = new cp[n];
    double arg = PI;
    for(int k = n>>1;k;k>>=1,arg*=0.5){
        cp  wm = std::polar(1.0,f*arg),w(1,0);
        for(int i = 0;i<n;i+=k,w*=wm){
            int p = i << 1;
            if(p>=n) p-= n;
            for(int j = 0;j<k;++j){
                b[i+j] = a[p+j] + w*a[p+k+j];
            }
        }
        for(int i = 0;i<n;++i) a[i] = b[i];
    }
    delete []b;
}

// use to check fft is right
void dft(cp *a,int n,int f)
{
    cp *b = new cp[n];
    for(int i = 0;i < n;i++) {
        b[i] = cp(0, 0);

        for(int j = 0;j < n;j++) {
            b[i] += cp(std::real(a[j])*cos(-2.0*PI*j*i/n), std::real(a[j])*sin(-2.0*PI*j*i/n));
        }
    }
    for(int i = 0;i<n;++i) a[i] = b[i];

    delete []b;
}

// a's size should be more then 2*n
void dct(double *a,int n,int f)
{
    cp *b = new cp[2*n];
    for(int i = n-1;i >= 0;i--) {
        b[n-i-1] = b[n+i] = cp(a[i], 0);
    }
    dft(b, 2*n, f);

    for(int i = 0;i < 2*n;i++)
        a[i] = std::real(b[i]);
    delete [] b;
}
void dct2(double *a, int n) {
    double *b = new double[n];
    for(int i = 0;i < n;i++) {
        b[i] = 0.0;
        for(int j = 0;j < n;j++) 
            b[i] += a[j] * cos(PI*i*(j+1.0/2)/n);
    }
    for(int i = 0;i < n;i++)
        a[i] = b[i] * sqrt(2.0/n) / sqrt(2.0);
    delete [] b;
}

// -log(x+y)  a = -log(x) b = -log(y)
double logInsideSum(double a, double b) {
    if(a >= b) std::swap(a, b);
//    printf("%lf %lf %lf\n", a, b,a - log(1.0 + pow(e, a-b)));
    return a - log(1.0 + pow(e, a-b));
}

// -log((abs(x-y))  a = -log(x) b = -log(y)
double logInsideDist(double a, double b) {
    if(a >= b) std::swap(a, b);
//    printf("%lf %lf %lf\n", a, b,a - log(1.0 + pow(e, a-b)));
    return a - log(1.0 - pow(e, a-b));
}

// probability  to cost
double p2cost(double p) {
    if(p <= 0) return Feature::IllegalDist;
    return - log(p);
}

double cost2p(double cost) {
    return pow(e, -cost);
}

void matrix2vector(const Matrix<double> & data, double *vec){
    int rowSize=data[0].size();
    for(int i=0; i<data.size(); i++){
        for(int j=0; j<rowSize; j++){
            *vec = data[i][j];
            vec++;
        }
    }
}

void vector2matrix(double *vec, Matrix<double> & data){
    for(int i=0; i< data.size(); i++){
        for(int j=0; j<data[0].size(); j++){
            data[i][j] = *vec;
            vec++;
        }
    }    
}
