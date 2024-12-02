import logging
from tqdm import tqdm

from brainscore.metrics import Score
from neural_nlp import models
from neural_nlp.benchmarks import benchmark_pool
from neural_nlp.models import get_activations, model_layers, model_pool, SubsamplingHook
from neural_nlp.neural_data.fmri import load_rdm_sentences as load_neural_rdms, load_voxels
from result_caching import store
from neural_nlp import FixedLayer

from neural_nlp_custom import load_model

# from neural_nlp.models.wrapper.core import ActivationsExtractorHelper
# from neural_nlp.models.wrapper.pytorch import PytorchWrapper

_logger = logging.getLogger(__name__)

@store(identifier_ignore=['base_model','presaved','weight_config','layers', 'prerun', 'base_model_impl'])
def score(benchmark, model, base_model, presaved=None, weight_config=None, layers=None, base_model_impl=None, subsample=None):
    model_impl = load_model(model=model, base_model=base_model, presaved=presaved, weight_config=weight_config)
    layers = layers or model_layers[base_model]
    if subsample:
        SubsamplingHook.hook(base_model, subsample)
    
    _logger.info('Loading benchmark')
    benchmark_impl = benchmark_pool[benchmark]

    _logger.info('Running')
    # shortcut for performance benchmarks
    if any(benchmark.startswith(performance_prefix) for performance_prefix in ['wikitext', 'glue']):
        return benchmark_impl(model_impl)

    # only last layer for behavioral benchmarks
    if benchmark.startswith('Futrell2018'):
        layers = layers[-1:]

    layer_scores = []
    for i, layer in enumerate(tqdm(layers, desc='layers')):
        candidate = FixedLayer(model_impl, layer, prerun=layers if i == 0 else None)  # prerun everything for 1st layer
        layer_score = benchmark_impl(candidate)
        layer_score = layer_score.expand_dims('layer')
        layer_score['layer'] = [layer]
        layer_scores.append(layer_score)
    layer_scores = Score.merge(*layer_scores)
    layer_scores = layer_scores.sel(layer=layers)  # preserve layer ordering
    layer_scores.attrs['model'] = model
    layer_scores.attrs['benchmark'] = benchmark
    return layer_scores

