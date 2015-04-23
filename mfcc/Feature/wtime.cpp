
#include "wtime.h"

double wtime(void)
{
    double          now_time;
    struct timeval  etstart;
    struct timezone tzp;

    if (gettimeofday(&etstart, &tzp) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((double)etstart.tv_sec) +              /*  in seconds */
        ((double)etstart.tv_usec) / 1000000.0;  /*  in microseconds */
    return now_time;
}

