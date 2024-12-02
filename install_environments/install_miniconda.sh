#!/bin/bash
# install_miniconda.sh

# make sure curl is installed
sudo apt install curl -y

# download miniconda with python 3.10
curl https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# install miniconda 
bash Miniconda* -u
conda update conda

# install jupyterlab
conda install -n base -c conda-forge jupyterlab jupyterlab_widgets
conda init

