import sys
from .utils.my_utils import *
from .utils.ProjectManager import ProjectManager
PM = ProjectManager()

import pandas as pd

def get_sentence_data(activations_index=0, apai=500):
    """
    apai is the number of sentences per activation index
    """
    i1=apai*activations_index
    i2=apai*(activations_index+1)
    df = PM.load_dataset("gpt_input")
    df=df[(i1 <= df["sentence_idx"]) & (df["sentence_idx"] < i2)]
    sentences_in = [list(x[1]) for x in df.groupby("sentence_idx")["model_words"]]
    append_spaces_in = [list(x[1]) for x in df.groupby("sentence_idx")["append_space"]]
    return sentences_in, append_spaces_in


