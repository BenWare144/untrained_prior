#!/usr/bin/env python
# coding: utf-8

# Init
import torch
print(torch.__file__)
import numpy as np
print(torch.cuda.is_available())
import gc
import os
HOME=os.getenv("HOME")

import copy

import sys
if not f"{HOME}/untrained_prior/jupyterlab/utils" in sys.path: sys.path.append(f"{HOME}/untrained_prior/jupyterlab/utils")

from my_utils import print_dimensions, print_layer_names
from transformers import GPT2Tokenizer, GPT2Model, AutoTokenizer, AutoConfig

model_label = "gpt2-untrained"
model_dir = f"{HOME}/data/transformer_saved_models/"
weight_dir = f"{HOME}/data/transformer_weights/"
model_name = f"gpt2-untrained_1"

# Load initial untrained_gpt2 into memory
model = GPT2Model.from_pretrained(os.path.join(model_dir, model_name),output_hidden_states=True) # .to("cuda")

### create some data to force the model to load weights
tokenizer = GPT2Tokenizer.from_pretrained(model_label.replace("-untrained",""))
text = "Replace me by any text you would like. And a second sentence can't."
encoded_input = tokenizer(text, return_tensors='pt') # .to("cuda")
output = model(**encoded_input)

# generate_weight_configs
def generate_weight_configs(model, model_label ,weight_dir=weight_dir,v=0):
    HF_initial_weights = copy.deepcopy(model.h[0].state_dict())

    # these two values should not be touched
    del HF_initial_weights["attn.bias"]
    del HF_initial_weights["attn.masked_bias"]

    np.random.seed(1)
    normal_initial_weights={k:torch.from_numpy(np.random.normal(loc=0.0, scale=0.2, size=v.shape)) for k,v in HF_initial_weights.items()}

    # create weight configs
    names=list(normal_initial_weights)
    weight_configs={}
    for i,x in enumerate(list(normal_initial_weights)):
        label=f"{model_label}_weight_config_single_{i}"
        weights = {x:normal_initial_weights[x]}
        weight_configs[label] = weights
    for i in range(int(len(normal_initial_weights)/2)):
        label=f"{model_label}_weight_config_double_{i}"
        weight_configs[label] = {
            names[2*i]: normal_initial_weights[names[2*i]],
            names[2*i+1]: normal_initial_weights[names[2*i+1]],
        }
    label=f"{model_label}_weight_config_quad_attn"
    weight_configs[label] = {k:v for k,v in normal_initial_weights.items() if "attn.c" in k}
    label=f"{model_label}_weight_config_quad_mlp"
    weight_configs[label] = {k:v for k,v in normal_initial_weights.items() if "mlp.c" in k}
    label=f"{model_label}_weight_config_quad_ln"
    weight_configs[label] = {k:v for k,v in normal_initial_weights.items() if "ln_" in k}
    label=f"{model_label}_weight_config_all"
    weight_configs[label] = normal_initial_weights
    for weights_name, weight_config in weight_configs.items():
        if v:
            print(f"    {weights_name}:")
            for k in list(weight_config):
                if k in list(normal_initial_weights):
                    print(f"{k}: ", end="")
                    # if x in list(weight_config):
                    print_dimensions(weight_config[k])
        torch.save(weight_config, os.path.join(weight_dir, weights_name+".pt"))
    return weight_configs

weight_configs = generate_weight_configs(model,model_label,v=1)

