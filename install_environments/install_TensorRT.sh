#!/bin/bash
# install_TensorRT.sh

##################
### Install TensorRT
##################
# TensorRT 8.5.3
# TensorRT install guide: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-853/install-guide/index.html
# download TensorRT 8.5.3 (TensorRT 8.5 GA Update 2) from here: https://developer.nvidia.com/nvidia-tensorrt-8x-download
# direct link: https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb
# nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb

# Requires pyCUDA, which is installed in the untrained_prior conda environment
if [ $CONDA_DEFAULT_ENV != "untrained_prior" ]; then conda activate untrained_prior; fi

cd ~/Downloads/Cuda_Files
os="ubuntu2204"
tag="8.5.3-cuda-11.8"
sudo dpkg -i nv-tensorrt-local-repo-${os}-${tag}_1.0-1_amd64.deb
sudo cp /var/nv-tensorrt-local-repo-${os}-${tag}/*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install tensorrt

### might need to run this
# python3 -m pip install numpy

sudo apt-get install python3-libnvinfer-dev

### might need to run this
# python3 -m pip install protobuf

sudo apt-get install uff-converter-tf

# Verify
# dpkg-query -W tensorrt
# output should be something like:
## tensorrt	8.5.3.1-1+cuda11.8
