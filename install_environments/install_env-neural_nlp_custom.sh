#!/bin/bash
# install_env-neural_nlp_custom.sh

# create environment from the yaml file
conda env create -f ~/untrained_prior/install_environments/env-neural_nlp_custom.yaml

# Activate environment
if [ $CONDA_DEFAULT_ENV != "neural_nlp_custom" ]; then conda activate neural_nlp_custom; fi

# install backdated packages github repos
pip install dask[array]==2.1.0
pip install git+https://github.com/mschrimpf/lm_1b.git@1ff7382 # - lm-1b==0.0.1
pip install git+https://github.com/mschrimpf/OpenNMT-py.git@f339063 # - opennmt-py==0.2
pip install git+https://github.com/mschrimpf/skip-thoughts.git@c8a3cd5 # - skip-thoughts==0.0.1
pip install git+https://github.com/nltk/nltk_contrib.git@c9da2c2 # - nltk-contrib==3.4.1 (from here: https://github.com/nltk/nltk_contrib/tree/python3)

# install local repos that have been backdated and modified for compatabiltiy.
pip install -e ~/untrained_prior/neural-nlp-packages/brainio_base --no-deps
pip install -e ~/untrained_prior/neural-nlp-packages/brainio_collection --no-deps
pip install -e ~/untrained_prior/neural-nlp-packages/result_caching --no-deps
pip install -e ~/untrained_prior/neural-nlp-packages/brain-score --no-deps
pip install -e ~/untrained_prior/neural-nlp-packages/neural-nlp --no-deps
pip install -e ~/untrained_prior/neural-nlp-packages/neural-nlp-custom --no-deps

# If errors involving pickle occur, try installing pickle5
pip install pickle5 




















