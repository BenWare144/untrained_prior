## Installation

### Requirements
* OS: Ubuntu 22.04

### Cuda Installation
Note: Instructions are for ubuntu2204. Please see the comments in the installation scripts if installation on another OS is desired or problems occur.

1. Install Cuda 11.8.0
```bash
~/untrained_prior/install_environments/install_cuda_11_8.sh
```

2. Set environmental variables in the `~/.bashrc`
```bash
printf "\n\n%s\n%s\n" 'export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}' 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
export PATH=/usr/local/cuda-11.8/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

3. Install cuDNN 8.6.0
```bash
~/untrained_prior/install_environments/install_cuDNN_8_6.sh
```

4. (Optional) Verify Cuda installation
```bash
# Install Cuda samples
~/untrained_prior/install_environments/install_cuda_11_8_samples.sh

# Verify
~/Downloads/Cuda_Files/cuda-samples-11.8/bin/x86_64/linux/release/deviceQuery
```

5. (Optional) Verify cuDNN installation
```bash
# Install cuDNN samples
~/untrained_prior/install_environments/install_cuDNN_8_6_samples.sh

# Verify
~/Downloads/Cuda_Files/cudnn_samples_v8/mnistCUDNN/mnistCUDNN
```

6. (Optional) Verify versions
```bash
# NVIDIA Display Driver Version
nvidia-smi | grep "Driver Version" | awk '{print $6}' | cut -c1- # 550.107.02

# CUDA version
nvcc --version | grep "release" | awk -F'[, ]' '{print $6}' # 11.8

# CUDA Toolkit Version
nvcc --version | grep "release" | awk '{print $6}' # V11.8.89

# cuDNN Library Version
locate cudnn | grep "libcudnn.so." | tail -n1 | sed -r 's/^.*\.so\.//' # 8.6.0
```


### Install Conda Environments

This project uses two environments.
* neural_nlp_custom is required for extracting activations and reproducing the original scores from using backdated code from [Schrimpf (2020)](https://github.com/mschrimpf/neural-nlp). (Python 3.7 and Tensorflow 1.14.0)
* untrained_prior is the analysis environment required to run experiments.


1. Install conda if you do not already have conda installed. Here is a script for installing conda via miniconda.
```bash
~/untrained_prior/install_environments/install_miniconda.sh
```

2. Install the neural_nlp_custom conda environment.
```bash
~/untrained_prior/install_environments/install_env-neural_nlp_custom.sh
```

3. Install the untrained_prior conda environment.
```bash
~/untrained_prior/install_environments/install_env-untrained_prior.sh
```

4. Install Jupyter Lab hooks.
```bash
# In the "base" conda environment:
ipython kernel install --user --name neural_nlp_custom
ipython kernel install --user --name untrained_prior
```

### (Optional) Install TensorRT

Login to nvidia and download TensorRT from [direct link](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/local_repos/nv-tensorrt-local-repo-ubuntu2204-8.5.3-cuda-11.8_1.0-1_amd64.deb) or [this download page](https://developer.nvidia.com/nvidia-tensorrt-8x-download) and place it in `~/Downloads/Cuda_Files`. 

Then install with
```bash
# Install TensorRT 8.5.3
~/untrained_prior/install_environments/install_TensorRT.sh

# Verify
dpkg-query -W tensorrt # tensorrt 8.5.3.1-1+cuda11.8
```

## Testing environment
Extraction of model activations was performed on:
* OS: Ubuntu 22.04
* GPU: NVIDIA TITAN Xp
* Cuda: 11.8.0
* cuDNN: 8.6.0

Brain scores for pretrained models were reproduced to 4 significant figures. The only exception were the scores for the models' the 0th (dropout) layer, which I was unable to reproduce. Brain scores were also calculated in the original stated environment (Ubuntu 16.04, Cuda 10.1, and cuDNN 7.6) but yielded the same results.




