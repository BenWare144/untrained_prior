import logging
from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store
from neural_nlp import FixedLayer

# from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
# from neural_nlp.models.wrapper.pytorch import PytorchWrapper


from neural_nlp_custom.custom_utils import get_sentence_data
from neural_nlp.models.implementations import word_last
import numpy as np
import pandas as pd

import sys

from .utils.ben_utils import *
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
    # save_activations_to_file(activations, layers, model, version, activations_index, metadata_df)

    # model = model_ctr.from_pretrained("gpt2", output_hidden_states=True, state_dict=state_dict)


    # model_impl._model = model
    # # model_impl._model.eval()
    # model_impl.content.__class__ = ModifiedPytorchTransformerWrapper
    # model_impl._model_container.__class__ = ModifiedPytorchTransformerWrapper.ModelContainer
    # model_impl._model_container = model_impl.ModelContainer(tokenizer, model_impl._model, layers, tokenizer_special_tokens)
    # model_impl._extractor = ActivationsExtractorHelper(identifier=model, get_activations=model_impl._model_container, reset=lambda: None)
    # model_impl._extractor.insert_attrs(model_impl)
    # print(f"reinitialized model={model}, num_layers={len(layers)}, layers: {layers}")





    # _logger.info('Loading benchmark')
    # benchmark_impl = benchmark_pool[benchmark]

    # _logger.info('Running')
    # # shortcut for performance benchmarks
    # if any(benchmark.startswith(performance_prefix) for performance_prefix in ['wikitext', 'glue']):
    #     return benchmark_impl(model_impl)

    # # only last layer for behavioral benchmarks
    # if benchmark.startswith('Futrell2018'):
    #     layers = layers[-1:]

    # layer_scores = []
    # for i, layer in enumerate(tqdm(layers, desc='layers')):
    #     candidate = FixedLayer(model_impl, layer, prerun=layers if i == 0 else None)  # prerun everything for 1st layer
    #     layer_score = benchmark_impl(candidate)
    #     layer_score = layer_score.expand_dims('layer')
    #     layer_score['layer'] = [layer]
    #     layer_scores.append(layer_score)
    # layer_scores = Score.merge(*layer_scores)
    # layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    # layer_scores.attrs['model'] = model
    # layer_scores.attrs['benchmark'] = benchmark
    # return layer_scores



# ========================
# ========================
# ========================
    # base_model = model if model in model_pool.keys() else "gpt2-untrained"
    # # original code
    # base_model_impl = base_model_impl or model_pool[base_model]
    # if subsample:
    #     SubsamplingHook.hook(base_model, subsample)
    # layers = layers or model_layers[base_model]

    # base_model_impl._model.to("cpu")

    # # if not model in model_pool.keys():
    # # return model_impl, layers
    # # identifier=model
    # # del model  ########################
    # saved_model_loc=f"/home/ben/data/transformer_saved_models/E1_Schrimpfs_{base_model}"
    # print(f"loading presaved model {model} from {saved_model_loc}")

    # from transformers import GPT2Model
    # my_model = GPT2Model.from_pretrained(saved_model_loc, output_hidden_states=True)

    # if base_model != model:
    #     load_weight_config_from_local_file(my_model, file_name=f"{model}.pt")


    # model_impl = ModifiedPytorchTransformerWrapper(
    #     identifier = model,
    #     tokenizer = base_model_impl._tokenizer,
    #     tokenizer_special_tokens = base_model_impl._model_container.tokenizer_special_tokens,
    #     model = my_model,
    #     layers = layers,
    #     sentence_average = word_last)
    # del base_model_impl
