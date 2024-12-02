#!/bin/bash
# install_cuDNN_8_6_samples.sh

cd ~/Downloads/Cuda_Files
cp -r /usr/src/cudnn_samples_v8/ .
cd ~/Downloads/Cuda_Files/cudnn_samples_v8/mnistCUDNN
make clean && make

