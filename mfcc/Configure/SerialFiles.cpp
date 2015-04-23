// =====================================================================================
// 
//       Filename:  SerialFiles.cpp
// 
//    Description:  
// 
//        Version:  0.01
//        Created:  2014/10/17 11时52分01秒
//       Revision:  none
//       Compiler:  clang 3.5
// 
//         Author:  wengsht (SYSU-CMU), wengsht.sysu@gmail.com
//        Company:  
// 
// =====================================================================================

#include "SerialFiles.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <stdarg.h>
#include "configure_hmm.h"

#define DEFAULT_SIZE_BIT 2

const int SerialFiles::maxSizeBit = 13;


SerialFiles::SerialFiles() {
    sizeBit = DEFAULT_SIZE_BIT;

//    printf("%d\n", inAlise.size());

//    loadAlise();
}

SerialFiles::~SerialFiles() {
}

void SerialFiles::setSizeBit(int sizeBit) {
    this->sizeBit = sizeBit;
}

// "1" ----> "one"
const char * SerialFiles::inAlias(char * in) {
#define ALISE(a,b) \
    if(strcmp(in, a) == 0) \
        return b;
#include "alias.def"
#undef ALISE
    return in;
}

// "one" ---> "1"
const char * SerialFiles::outAlias(char * out) {
#define ALISE(a,b) \
    if(strcmp(out, b) == 0) \
        return a;
#include "alias.def"
#undef ALISE
    return out;
}

void SerialFiles::getSerialFileName(char * out, int seqNum, int prefixNum, ...) {
    va_list pvar;
    va_start(pvar, prefixNum);
    out[0] = 0;
    while(prefixNum --) {
        if(out[0]) 
            strcat(out, "_");

        const char * prefix = outAlias(va_arg(pvar, char *));

        strcat(out, prefix);
    }
    va_end(pvar);

    strcat(out, "_");
    strcat(out, fullfill(seqNum));
}

char *SerialFiles::fullfill(int num) {
    static char tmp[maxSizeBit];
    int idx = sizeBit - 1;
    tmp[sizeBit] = 0;

    memset(tmp, '0', sizeBit);
    while(num) {
        tmp[idx --] = '0' + num % 10;
        num /= 10;
    }
    return tmp;
}

void SerialFiles::parseSerialFileName(const char * const fileName, int &seqNum, int prefixNum, ...) {
    int idx, len = strlen(fileName);
    va_list pvar;

    char * tmp = new char[len+1];
    strcpy(tmp, fileName);

    va_start(pvar, prefixNum);

    char *erase = strstr(tmp, ".");
    if(erase) 
        erase[0] = NULL;

    char *token = strtok(tmp, "_");
    while(prefixNum -- && token) {
        char *prefix = va_arg(pvar, char *);

        const char *tmp = inAlias(token);

        strcpy(prefix, tmp);

        token = strtok(NULL, "_");
    }
    if(token)
        seqNum = atoi(token);
    else 
        seqNum = 1;

    va_end(pvar);

    delete [] tmp;
}

bool SerialFiles::isNotWord(char *word) {
    if(strcmp(word, TROPE TROPE_SILENCE) == 0)
        return true;
    return false;

}
