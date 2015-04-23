# Term Project for 18652 <How to Write Fast Code>

`@Shitao Weng`: sweng@andrew.cmu.edu
`@Shushan Chen`: shushanc@andrew.cmu.edu

## Preprequisite 

`CUDA 6.5`
`CMake 2.8+`
`g++ (c++11 stardard)`

## Install 
    
    git clone .
    cd cuda-mfcc-gmm/
    mkdir build/
    cmake ../
    make && make install
    
## Run

    cd cuda-mfcc-gmm/RunEnv
    
### MFCC Extraction

#### CPU 

    #./cpu_mfcc -l wav_file_name_without_prefix
    ./cpu_mfcc -l SW_20001_ch2_cut
    
Stardard Output Stream:

    PreEmp: 1.0210488 s , 1.36219%
    Windowing: 0.0832372 s , 5.38676%
    FFT padding: 0.018985 s , 1.22863%
    PowerSpectrum: 0.586775 s , 37.9736%
    MelFiltering: 0.434919 s , 28.1461%
    DCT Ceptrum: 0.39311 s , 25.4404%
    Normalization: 0.00714397 s , 0.462328%
    
MFCC feature will be stored in file `normalMelCeps.txt`.
    
#### CUDA 

    #./cuda_mfcc -l wav_file_name_without_prefix
    ./cuda_mfcc -l SW_20001_ch2_cut
    
Stardard Output Stream:
    
    CUDA Initialize Time: 2.32246
    Total Time (Without InitializeTime) : 0.070796
    PreProcessing: 0.00680399 s , 9.6107%
    FFT: 0.051362 s , 72.5493%
    MelFiltering: 0.00560689 s , 7.91978%
    DCT Ceptrum: 0.00620914 s , 8.77046%
    Normalization: 0.000813961 s , 1.14973%
    
MFCC feature will be stored in file `cuda_normalMelCeps.txt`.

Comparing the running of each stages used by CPU and GPU, we get a 25.5x speedup without considering the CUDA initialization time. 

#### Verify the results

    diff normalMelCeps.txt cuda_normalMelCeps.txt

### GMM Training (Small test)
        
#### CPU GMM Training

    # ./gmm_train_main config_file ouput_model_file_name
    ./gmm_train_main ubm.64.cfg ubm.cpu.64.model
    
##### Stardard Output Stream

    *************Split MixNum = 32 Finished ******************
    Avg EM Iteration: 0.768752
    Avg EM time 2.306273
    Avg MixUp time 64, 0.000240
    Avg KMean time 0.843861
    Avg KMean Iteration: 0.281287
    *************Split MixNum = 64 Finished ******************
    Avg EM Iteration: 1.553885
    Avg EM time 18.646647
    all training has been finished
    Last Gmm MixtureNum=64
    Whole runtime : 25.036303
#### CUDA GMM Training

    # ./cuda_gmm_train_main config_file ouput_model_file_name
    ./cuda_gmm_train_main ubm.64.cfg ubm.cuda.64.model
    
##### Stardard Output Stream

    ...
    *************Split MixNum = 32 Finished ******************
    Avg EM Iteration: 0.039880
    Avg EM time 0.119680
    Avg MixUp time 64, 0.000319
    Avg KMean time 0.004120
    Avg KMean Iteration: 0.001373
    *************Split MixNum = 64 Finished ******************
    Avg EM Iteration: 0.070802
    Avg EM time 0.849650
    all training has been finished
    Last Gmm MixtureNum=64
    Whole runtime : 3.408745
    
##### Verify the result

On this small test, we get `7x` speedup(100 wav files, 21420 frames). If we test it on a big data set (4000 wav files, 971526 frames), we can get `27x` speed up.

### Spoofing Countermeasure (GMM Training Big Test)

Before you start running this, please download the mfcc features files on dropbox (`mfcc.zip`).

    cd cuda-mfcc-gmm/RunEnv
    unzip mfcc.zip
    ./cuda_gmm_train_main ubm.512.cfg  ubm.cuda.512.model
    ./svm_feature_extract ubm.cuda.512.model all.list all.tags 16375
    
This will generate two libsvm feature files `svm_train.feature` and `svm_test.feature`.

After generating feature files for libsvm, we can use libsvm to classify the wav files as human utterance or spoofed utterance.

    ./svm-train -s 0 -t 1 -d 3 -b 1  -g 1  -r 1 svm_train.feature spoof.model
    ./svm-predict -b 1 svm_test.feature spoof.model result.txt
    
#### Stardard output

    Accuracy = 85.1083% (45424/53372) (classification)]
    
## License

The MIT License (MIT)

Copyright (c) <2015> Shitao Weng<sweng@andrew.cmu.edu>, Shushan Chen<shushanc@andrew.cmu.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
