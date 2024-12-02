import logging
from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store
from neural_nlp import FixedLayer


_logger = logging.getLogger(__name__)


def create_presaved_model(model, base_model, layers=None, model_impl=None, subsample=None):
    # base_model = model if model in model_pool.keys() else "gpt2-untrained"

    # original code
    model_impl = model_impl or model_pool[base_model]
    if subsample:
        SubsamplingHook.hook(base_model, subsample)
    layers = layers or model_layers[base_model]

    import os
    HOME=os.getenv("HOME")
    
    from pathlib import Path
    
    save_loc= str(Path(f"{HOME}/data/transformer_saved_models/{model}").expanduser())
    # Path(save_loc).mkdir(exist_ok=True)
    print("========================================")
    print("========================================")
    print("========================================")
    print(f"save loc={save_loc}")
    print(os.path.isdir(save_loc))
    os.mkdir(save_loc) 
    print(os.path.isdir(save_loc))
    print("========================================")
    print("========================================")
    print("========================================")
    model_impl._model.save_pretrained(save_loc)
    print("="*25)

