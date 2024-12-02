#!/bin/bash
# install_cuda_11_8_samples.sh

cd ~/Downloads/Cuda_Files
wget -qO- https://github.com/NVIDIA/cuda-samples/archive/refs/tags/v11.8.tar.gz | tar xvz
make -C ~/Downloads/Cuda_Files/cuda-samples-11.8 

