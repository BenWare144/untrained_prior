
import argparse
import fire
import logging
import sys
import os
HOME=os.getenv("HOME")

from datetime import datetime

# from neural_nlp_score import score as score_function

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()

def score(benchmark, model, base_model, presaved=None, weight_config=None, layers=None, subsample=None):
    os.environ["RESULTCACHING_HOME"] = f"{HOME}/.result_caching/{model}"
    from neural_nlp_custom import score as score_function
    setup_logging()
    # print("========================================================")
    # print("========================================================")
    # print("========================================================")
    # print(os.environ["RESULTCACHING_HOME"])
    # print("========================================================")
    # print("========================================================")
    # print("========================================================")
    

    start = datetime.now()
    time_stamp=start
    scores_fn=f"{HOME}/data/scores/{model}_{benchmark}_{time_stamp}_score_raw"
    print("scores_fn:",scores_fn)

    score = score_function(benchmark=benchmark, model=model, base_model=base_model, presaved=presaved, weight_config=weight_config, layers=layers, subsample=subsample)
    end = datetime.now()
    print(f"Duration: {end - start}")
    print(score)
    with open(f"{scores_fn}.txt", 'w') as f:
        f.write(str(score))
    df=score.to_dataframe(name=f"{scores_fn}")
    df.to_csv(f"{scores_fn}.csv")
    
def get_activations(model, base_model, presaved=None, weight_config=None, layers=None, subsample=None):
    from neural_nlp_custom import record_activations as record_activations_function
    setup_logging()

    start = datetime.now()
    record_activations_function(model=model, base_model=base_model, presaved=presaved, weight_config=weight_config, layers=layers, subsample=subsample)
    end = datetime.now()
    print(f"Duration: {end - start}")


def create_presaved(model, base_model, layers=None, subsample=None):
    from neural_nlp_custom import create_presaved_model as create_function
    setup_logging()

    start = datetime.now()
    create_function(model=model,base_model=base_model, layers=layers, subsample=subsample)
    end = datetime.now()
    print(f"Duration: {end - start}")





def setup_logging():
    logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
    _logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")
    for ignore_logger in ['transformers.data.processors', 'botocore', 'boto3', 'urllib3', 's3transfer']:
        logging.getLogger(ignore_logger).setLevel(logging.INFO)


if __name__ == '__main__':
    import warnings

    warnings.simplefilter(action='ignore', category=FutureWarning)
    # warnings.simplefilter(action='ignore', category=DeprecationWarning)

    fire.Fire(command=FIRE_FLAGS)


### warnings are a part of the tf package
# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'