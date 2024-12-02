#!/bin/bash
# install_cuda_11_8.sh

##################
### CUDA background
##################
# GPU			Compute Capability
# NVIDIA TITAN Xp	6.1

### Tensorflow compatability reference
# Version            Python version  Compiler   Build tools  cuDNN  CUDA  TensorRT 
# tensorflow-2.12.0  3.8-3.11        GCC 9.3.1  Bazel 5.3.0  8.6    11.8  8.5.3

##################
### Install CUDA toolkit
##################
# CUDA toolkit: 11.8.0 (Linux Ubuntu 22.04 x86_64)
# from here: https://www.nvidia.com/Download/driverResults.aspx/204639/en-us/

sudo apt-get -y install linux-headers-$(uname -r)
cd ~/Downloads/Cuda_Files
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
# optional libraries
sudo apt-get install g++ freeglut3-dev build-essential libx11-dev \
    libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev
    

# conda activate neural_nlp_custom
# python3 ~/untrained_prior/scripts/E0_1_gen_weight_configs.py










