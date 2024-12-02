#!/bin/bash
# neural-nlp-scripts.sh
# if [ $CONDA_DEFAULT_ENV != "neural_nlp_custom" ]; then conda activate neural_nlp_custom; fi

function create_model() {(
    model=$1
    base_model=$2

    echo =======================================
    echo =======================================
    echo =======================================
    echo model:         $model
    echo base_model:    $base_model
    echo CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV
    echo =======================================
    echo =======================================
    echo =======================================

    python -m neural_nlp_custom create_presaved --model $model --base_model $base_model --log_level DEBUG
)}

function score_model() {(
    # benchmarks: Blank2014fROI-encoding Fedorenko2016v3-encoding Pereira2018-encoding
    benchmark=$1
    model=$2
    base_model=$3
    presaved="${4:-None}"
    weight_config="${5:-None}"

    echo =======================================
    echo =======================================
    echo =======================================
    echo benchmark:     $benchmark
    echo model:         $model
    echo base_model:    $base_model
    echo presaved:      $presaved
    echo weight_config: $weight_config
    echo CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV
    echo =======================================
    echo =======================================
    echo =======================================

    python -m neural_nlp_custom score --model $model --benchmark $benchmark --base_model $base_model --presaved $presaved --weight_config $weight_config --log_level DEBUG
)}

function get_activations() {(
    model=$1
    base_model=$2
    presaved="${3:-None}"
    weight_config="${4:-None}"
    
    echo =======================================
    echo =======================================
    echo =======================================
    echo model:         $model
    echo base_model:    $base_model
    echo presaved:      $presaved
    echo weight_config: $weight_config
    echo CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV
    echo =======================================
    echo =======================================
    echo =======================================
    

    python -m neural_nlp_custom get_activations --model $model --base_model $base_model --presaved $presaved --weight_config $weight_config --log_level DEBUG
)}
