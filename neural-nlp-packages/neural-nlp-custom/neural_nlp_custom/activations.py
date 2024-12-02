import logging
from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store
from neural_nlp import FixedLayer

from neural_nlp_custom.custom_utils import get_sentence_data
from neural_nlp.models.implementations import word_last
import numpy as np
import pandas as pd

import sys

from .utils.my_utils import *
from .utils.ProjectManager import ProjectManager
PM = ProjectManager()

_logger = logging.getLogger(__name__)

from neural_nlp_custom import load_model

def record_activations(model, base_model, presaved=None, weight_config=None, layers=None, base_model_impl=None, subsample=None):
    model_impl = load_model(model=model, base_model=base_model, presaved=presaved, weight_config=weight_config, for_activations=True)
    layers = layers or model_layers[base_model]

    version=1
    activations_index=0
    # if are_activations_already_saved(model,version,activations_index):
    #     print(f"activations already saved for model {model}, version={version}, activations_index={activations_index}")
    #     return
    
    print(f"getting activations for model {model}, version={version}, activations_index={activations_index}")
    sentences_in, append_spaces_in = get_sentence_data(activations_index) # to test: , apai=50

    # model_impl, layers = initialize_model_impl(model)
    word_encodings, metadata_df = model_impl._model_container(sentences=sentences_in, append_spaces=append_spaces_in, layers=layers, v=0)
    activations = np.concatenate([np.concatenate([word_encodings[layer][i] for i in range(len(word_encodings[layer]))], axis=1) for layer in layers],axis=0)

    # save activations
    PM.save_activations(activations, layers, model, metadata_df)
