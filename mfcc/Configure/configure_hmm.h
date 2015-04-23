#define AUTOMATON_STATE_NUM 5
#define GAUSSIAN_NUM 4

#define MAX_TRAIN_TIMES 100

#define DTW_MAX_FORWARD 3

// skip transition probability floor
#define FLOOR_TRANSITION_PROBABILITY 1e-24

#define TROPE "#"
#define TROPE_SILENCE "silence"
#define TROPE_PENALTY "penalty"

// 1+2+3  mean utterance 1->2->3
#define LINK_WORD "+"
#define TRAIN_INNER_PENALTY 1.0

// init kmean transfer probability 0.5+0.5 = 1.0!
#define INIT_KMEAN_0_1 0.5
#define INIT_KMEAN_0_2 0.5

#define INIT_MODEL_SUFFIX ".inithmm" ///< kmean_5_4_5.hmm.inithmm is used to init the training of it
