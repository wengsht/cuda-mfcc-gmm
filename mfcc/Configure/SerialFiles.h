// =====================================================================================
// 
//       Filename:  SerialFiles.h
// 
//    Description:  Generate serial file names format
//                  e.g. getSerialFileName("haha", 2, out) 
//                  out = haha_0002
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
#ifndef _AUTOGUARD_SerialFiles_H_
#define _AUTOGUARD_SerialFiles_H_

#include "unistd.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <cstring>
#include <vector>
#include <string>
#include "configure_basic.h"
#include <map>

class SerialFiles {
    public:
        SerialFiles();
        ~SerialFiles();

        void setSizeBit(int sizeBit);

        // out[] = prefix1_prefix2_prefixn_0000seqNum
        void getSerialFileName(char * out, int seqNum, int prefixNum, ...);

        // prefix_0004.wav  --> prefix = "prefix" seqNum = 4
        static void parseSerialFileName(const char * const fileName, int &seqNum, int prefixNum, ...);

//        static void load

        static bool isWavFile(char *fileName) {
            char * wavDot = strstr(fileName, WAV_SUFFIX);
            return wavDot && 0 == strcmp(wavDot, WAV_SUFFIX);
        }
        static std::string getMfccFileName(std::string wavFileName) {
            int idx = wavFileName.find(WAV_SUFFIX);
            if(idx == -1)
                return "";

            return wavFileName.replace(idx, wavFileName.length() - 1, MFCC_SUFFIX);
        }
        static void getWavFileNames(char *dir, std::vector<std::string> &wavFileNames) {
            static char buf[1024];
            DIR *dirp;
            struct dirent *file;
            struct stat fileStat;

            dirp = opendir(dir);

            while(NULL != (file = readdir(dirp))) {
                if(isWavFile(file->d_name)) {
                    wavFileNames.push_back(std::string(file->d_name));
                }
            }
            closedir(dirp);
        }

        // "1" ----> "one"
        static const char * inAlias(char * in);
        // "one" ---> "1"
        static const char * outAlias(char * out);

        static bool isNotWord(char *word);

    private:
        char *fullfill(int num);

        const static int maxSizeBit;


        // e.g fileName_xxxx.type 
        // sizeBit = 4
        int sizeBit;

};

#endif
