#!/bin/bash
# install_env-untrained_prior.sh

## installs the untrained_prior conda environment 
# includes
# - tensorflow with gpu
# - pytorch with gpu
# - transformers (huggingface)

conda env create -f ~/untrained_prior/install_environments/env-untrained_prior.yaml

conda activate untrained_prior
    pip install pip==23.1.2
    pip install --no-deps -r ~/untrained_prior/install_environments/env-untrained_prior-requirements.txt
conda deactivate
