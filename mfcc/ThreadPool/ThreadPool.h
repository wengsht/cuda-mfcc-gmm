#ifndef _AUTOGUARD_ThreadPool_H_
#define _AUTOGUARD_ThreadPool_H_

#include <unistd.h>
#include <pthread.h>
#include <vector>

typedef void (*TASK_FUNC)(void *) ;

struct sp_task {
    /*  func should be responsible for the free of in */
    TASK_FUNC func;
    void *in;
};
struct thread_info {
    std::vector<sp_task> tasks;

    pthread_t tid;
};

class ThreadPool {
    public:
        static int thread_num;
        ThreadPool();
        ThreadPool(int threadNum);
        ~ThreadPool();

        void run();

        void clear();

        void addTask(struct sp_task & task);

    private:
        static inline void *processThread(void *);
        std::vector< thread_info > threadTasks;

        void increament() {
            idx ++;
            if(idx >= threadNum)
                idx = 0;
        }
        
        int threadNum;

        int idx;
};

#endif

