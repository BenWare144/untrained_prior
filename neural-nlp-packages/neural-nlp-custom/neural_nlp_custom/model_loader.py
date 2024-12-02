import numpy as np
import pandas as pd

from transformers import GPT2Model
from neural_nlp.models import model_layers, model_pool
from neural_nlp.models.implementations import word_last, _PytorchTransformerWrapper
from neural_nlp_custom import ModifiedPytorchTransformerWrapper

import os
HOME=os.getenv("HOME")

def load_model(model=None, base_model=None, presaved=None, weight_config=None,for_activations=False, model_impl=None, layers=None):
    print(f"Loading Model: {model}")
    print(f"presaved: {presaved}")
    
    # base_model
    model_impl = model_impl or model_pool[base_model]
    layers = layers or model_layers[base_model]

    # abort if no presave
    if not presaved and not weight_config:
        return model_pool[base_model]
    
    # presaved
    if presaved:
        wrapper_dict={
            "identifier": model,
            "tokenizer": model_impl._tokenizer,
            "tokenizer_special_tokens": model_impl._model_container.tokenizer_special_tokens,
            "layers": model_impl.default_layers,
            "sentence_average":  word_last
        }
        model_impl._model.to("cpu")
        del model_impl

        presaved_model_loc=f"{HOME}/data/transformer_saved_models/{presaved}"
        # presaved_model_loc=f"{HOME}/data/transformer_saved_models/E1_Schrimpfs_{base_model}"
        print(f"Loading presaved model: {presaved} from {presaved_model_loc}")
        full_model = GPT2Model.from_pretrained(presaved_model_loc, output_hidden_states=True)

        # weight_config:
        if weight_config:
            print(f"Loading weight_config: {weight_config}")
            load_weight_config_from_local_file(full_model, file_name=f"{weight_config}.pt")
    
    # return
    if for_activations:
        return ModifiedPytorchTransformerWrapper(model = full_model,**wrapper_dict)
    else:
        return _PytorchTransformerWrapper(model = full_model,**wrapper_dict)

def load_weight_config_from_local_file(model, file_name=None, weight_dir=f"{HOME}/data/transformer_weights/"):
    import torch
    import os

    device = "cuda" if next(model.parameters()).is_cuda else "cpu"
    # device = "cuda"
    print("** load_weight_config_from_local_file start")
    print(f"**device={device}")
    print(f"**file_name={file_name}")
    print(f"**weight_dir={weight_dir}")

    label = file_name.split(".")[0]
    print(f"**label={label}")
    weights = torch.load(os.path.join(weight_dir, file_name), map_location=device)
    new_state_dict = {}
    for key, value in model.h[0].state_dict().items():
        if key in weights:
            new_state_dict[key] = weights[key].to(device)
        else:
            new_state_dict[key] = value.to(device)
    model.h[0].load_state_dict(new_state_dict)
    return label
