#!/bin/bash
# install_cuDNN_8_6.sh

##################
### Install cuDNN
##################
# cuDNN 8.6.0
# cuDNN Archive: https://developer.nvidia.com/rdp/cudnn-archive
# 8.6.0 Installation guide: https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-860/install-guide/index.html
# general cuDNN Installation guide: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#verify

cd ~/Downloads/Cuda_Files
wget https://developer.nvidia.com/compute/cudnn/secure/8.6.0/local_installers/11.8/cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2204-8.6.0.163_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install libcudnn8=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-dev=8.6.0.163-1+cuda11.8
sudo apt-get install libcudnn8-samples=8.6.0.163-1+cuda11.8

