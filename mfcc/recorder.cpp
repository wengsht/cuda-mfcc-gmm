#include <iostream>
#include "srs.h"
#include "SerialFiles.h"
#include "unistd.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include "Capture.h"
#include "EPAnalysis.h"
#include "AutoCapture.h"

using namespace std;

char dir[100] = TEMPLATES_DIR; //"./templates";

bool dealOpts(int argc, char **argv);
void captureRun(bool);

SerialFiles serialGenerator;

char user[100] = "anony";

bool playback = false;

int main(int argc, char **argv) {
    if( !dealOpts(argc, argv) ) 
        return 0;

    if(-1 == access(dir, F_OK))
        mkdir(dir, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    captureRun(playback);

    return 0;
}

bool dealOpts(int argc, char **argv) {
    int c;
    while((c = getopt(argc, argv, "hpd:u:")) != -1) {
        switch(c) {
            case 'h':
                printf("-p: playback\n \
                        -d dir: change templates dir\n \
                        -u username: 区分用户 \n");

                return false;
                break;
            case 'p':
                playback = true;
                break;
            case 'd':
                strcpy(dir, optarg);
                break;
            case 'u':
                strcpy(user, optarg);

                break;
        }
    }
    return true;
}
static int getBeginning(char *prefix) {
    static char buf[1024];
    DIR *dirp;
    struct dirent *file;
    struct stat fileStat;

    dirp = opendir(dir);

    int res = 0;
    int parseValue;
    char parseUser[100];
    while(NULL != (file = readdir(dirp))) {
        serialGenerator.parseSerialFileName(file->d_name, parseValue, 2,parseUser, buf);
        if((strcmp(buf, prefix) == 0 || strcmp(buf, SerialFiles::inAlias(prefix)) == 0 ) && strcmp(user, parseUser) == 0 && res < parseValue)
            res = parseValue;
    }
    closedir(dirp);

    return res + 1;
}

const char * stringFile(const char * a,const char * b,char *ab){
    ab[0] = '\0';
    strcpy(ab,dir);
    strcat(ab,"/");
    strcat(ab,a);
    strcat(ab,".");
    strcat(ab,b);
    return ab;
}

void captureRun(bool playback) {
    RawData data;
    DAEPAnalysis ep;
    AutoCapture capture(&ep);

    char buf[128];
    char fileName[128];
    char fn_buffer[128] = "";
    Tip("Input a word: ");
    while(cin >> buf) {
        int serialStart = getBeginning(buf);

        while(true) {
            Tip("Hit any key to capture new template, Hit <q> to start new word\n");
            int x = getch();
            if(x == 'q')
                break;

            if(capture.capture(&data)) {
                ep.smooth();
                ep.cut();

                serialGenerator.getSerialFileName(fileName, serialStart ++, 2, user, buf);

                cout << fileName << endl;

                data.saveWav(stringFile(fileName, "wav", fn_buffer));

                if(playback)
                    capture.play(&data);
            }
        }
        Tip("Input a new word[Ctrl-D to exit]: ");
    }
}
