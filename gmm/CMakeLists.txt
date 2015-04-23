set (MAIN gmm_train_main)
set (CUDAMAIN cuda_gmm_train_main)

set (TEST test_model)

set (SVM_FEATURE_EXTRA svm_feature_extract)

add_executable(${MAIN} main.cpp GMMParam.cpp GMMTrain.cpp config.cpp memory_srlr.cpp GMMMixup.cpp kmean.cpp Feat_comm.cpp wtime.cpp)

add_executable(${TEST} test_model.cpp GMMParam.cpp GMMTrain.cpp config.cpp memory_srlr.cpp GMMMixup.cpp kmean.cpp Feat_comm.cpp wtime.cpp)

cuda_compile(KMEAN_O kmean.cu)
cuda_compile(KMEAN_KERNEL_O kmean_kernel.cu)
cuda_compile(GMMTRAIN_O GMMTrain.cu)
cuda_compile(EM_KERNEL_O em_kernel.cu)

cuda_add_executable(${CUDAMAIN} main.cpp GMMParam.cpp ${GMMTRAIN_O} config.cpp memory_srlr.cpp GMMMixup.cpp ${KMEAN_O} Feat_comm.cpp wtime.cpp ${KMEAN_KERNEL_O} ${EM_KERNEL_O})

cuda_add_executable(${SVM_FEATURE_EXTRA} svm_feature_extract.cu GMMParam.cpp ${GMMTRAIN_O} config.cpp memory_srlr.cpp GMMMixup.cpp ${KMEAN_O} Feat_comm.cpp wtime.cpp ${KMEAN_KERNEL_O} ${EM_KERNEL_O})

#target_link_libraries(${MAIN} pthread)
 
install (TARGETS ${MAIN} DESTINATION ${PROJECT_SOURCE_DIR}/RunEnv)
install (TARGETS ${CUDAMAIN} DESTINATION ${PROJECT_SOURCE_DIR}/RunEnv)
install (TARGETS ${TEST} DESTINATION ${PROJECT_SOURCE_DIR}/RunEnv)
install (TARGETS ${SVM_FEATURE_EXTRA} DESTINATION ${PROJECT_SOURCE_DIR}/RunEnv)
