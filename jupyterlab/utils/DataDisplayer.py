import pandas as pd
import numpy as np
from pathlib import Path

import sys
import os
HOME=os.getenv("HOME")
if not f"{HOME}/untrained_prior/jupyterlab/utils" in sys.path: sys.path.append(f"{HOME}/untrained_prior/jupyterlab/utils")
from my_utils import *
from my_latex_tools import *

try:
    # print(get_ipython())
    from tqdm.notebook import tqdm
    # print("ipython enabled")
except:
    from tqdm import tqdm
    # print("no ipython")


class DataDisplayer:
    PM=None

    # models
    models = ['gpt2-xl', 'gpt2-xl-untrained_1', 'gpt2',
              *[f'gpt2-untrained_{i}' for i in range(1,10)],
              *[f'gpt2-untrained_{i}_weight_config_all' for i in range(1,10)],]
    model_names=unique(models)
    gpt2xl_models=models[:2]
    gpt2_models=models[2:]

    # labels
    labels = ['XL-Trained', 'XL-Untrained', 'GPT-Trained',
              *[f'GPT-Untrained {i}' for i in range(1,10)],
              *[f'GPT-Guassian {i}' for i in range(1,10)],]
    label_names=unique(labels)

    # model_to_modelIdx
    modelIdxs = list(range(len(models)))
    modelIdx_names=unique(modelIdxs)

    # groups
    groups = ["XL-Trained", "XL-Untrained", "GPT-Trained",
              *["GPT-Untrained"]*9,
              *["GPT-Guassian"]*9,]
    group_names=unique(groups)

    # groupIdxs
    groupIdxs = [1, 2, 3, *[4]*9, *[5]*9,]
    groupIdx_names=unique(groupIdxs)

    # ingroupIdxs
    ingroupIdxs = [1, 1, 1,
                   *[i for i in range(1,10)],
                   *[i for i in range(1,10)],]
    ingroupIdx_names=unique(ingroupIdxs)

    # linecolors
    # 'tab:blue' 'tab:orange' 'tab:green' 'tab:red' 'tab:purple' 'tab:brown' 'tab:pink' 'tab:gray' 'tab:olive' 'tab:cyan' # OLD: linecolor_names=['g', 'b', 'o', 'r', 'm']
    linecolors = ['tab:red', 'tab:orange', 'tab:green', *['tab:blue']*9, *['tab:purple']*9,]
    linecolor_names=unique(linecolors)

    # linestyles
    linestyle_names=[
        'solid',
        'dotted',
        'dashed',
        'dashdot',
        (0, (1, 4)), # 'loosely dotted',
        (0, (5, 5)), # 'dashed',
        (5, (10, 3)), # 'long dash with offset',
        (0, (3, 1, 1, 1)), # 'densely dashdotted',
        (0, (3, 3, 1, 3, 1, 3)), # 'dashdotdotted',
        ]
    linestyles=[*[linestyle_names[0]]*3,*linestyle_names*2]
    # ingroupIdx_to_linestyle=dict(zip(groupIdxs,linestyles))

    # model to X dicts
    model_to_label=dict(zip(models,labels))
    model_to_modelIdxs=dict(zip(models,modelIdxs))
    model_to_group=dict(zip(models,groups))
    model_to_groupIdx=dict(zip(models,groupIdxs))
    model_to_ingroupIdx=dict(zip(models,ingroupIdxs))
    model_to_linecolor = dict(zip(models,linecolors))
    model_to_linestyle = dict(zip(models,linestyles))
    group_to_linecolor = dict(zip(group_names,linecolor_names))
    group_to_groupIdx = dict(zip(group_names,groupIdx_names))
    
    # pos documentation: https://data.mendeley.com/datasets/zmycy7t9h9/2
    # where XX is for pos tag missing, and -LRB-/-RRB- is "(" / ")".
    # POS_51_all_tags=["XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
    #        "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
    #        "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", "WDT", "WP", "WP$", "WRB"] 
    # POS_12_all_tags=["VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X","."]
    # POS_7_all_tags=["Noun","Verb","Adposition","Determiner","Adjective","Adverb","X"]
    # POS_6_all_tags=["Noun","Verb","Adposition","Determiner","Adjective","Adverb"]
    # POS_12_to_POS_7={
    #     "NOUN":"Noun",
    #     "PRON":"Noun",
    #     "VERB":"Verb",
    #     "ADP":"Adposition",
    #     "DET":"Determiner",
    #     "ADJ":"Adjective",
    #     "ADV":"Adverb",
    #     "CONJ":"X",
    #     "NUM":"X",
    #     "PRT":"X",
    #     "X":"X",
    #     ".":"X"}

    layers_gpt2=["drop"]+[f'encoder.h.{i}' for i in range(12)]
    layers_gpt2_xl=["drop"]+[f'encoder.h.{i}' for i in range(48)]
    layer_label_to_idx={x:i for i,x in enumerate(layers_gpt2_xl)}
    layer_idx_to_label={i:x for i,x in enumerate(layers_gpt2_xl)}

    df_display_renameer={
        "benchmark":"Benchmark",
        "model":"Model",
        "model_group":"Model",
        "layer":"Layer",
        "layer_idx":"Layer",
        "target":"Target"
    }
    df_display_replacer={}
    display_model_names=model_to_label
    display_split_names={
        "Xss_6_train":"Training",
        "Xss_6_valid":"Validation",
        "Xss_6_test":"Testing",
        }
    display_benchmark_names={
        "Pereira2018-encoding": "Pereira2018",
        "Blank2014fROI-encoding": "Blank2014",
    }
    _display_target_names={
        "pos_tags": "POS-51",
        "POS_12_id": "POS-12",
        "POS_7_id": "POS-7",
        "sentence_idx": "Sentence Position",
        "unigram_probs": "Unigram Probabilities",
        "bigram_probs": "Bigram Probabilities",
        "trigram_probs": "Trigram Probabilities",
        "function": "Function vs Content",
        # "function": "Function",
        "tree_depth": "Tree Depth",
        "word_idx": "Word Order",
    }
    display_target_names={}
    for k,v in _display_target_names.items():
        display_target_names[k] = f"Current Word {v}"
        display_target_names[k+"-"] = f"Previous Word {v}"
        display_target_names[k+"+"] = f"Next Word {v}"
    
    display_POS_6_names={}
    def __init__(self, PM):
        # print(list(globals()))
        self.PM = PM
        self.classification_labels=PM.classification_labels
        self.reggression_labels=PM.reggression_labels
        self.pos_labels=PM.pos_labels
        # self.POS_7_map=
        self.POS_51_all_tags = PM.POS_51_all_tags
        self.POS_12_all_tags = PM.POS_12_all_tags
        self.POS_7_all_tags = PM.POS_7_all_tags
        self.POS_6_all_tags = PM.POS_6_all_tags
        self.POS_12_to_POS_7 = PM.POS_12_to_POS_7
        
        self.POS_51_tag_to_id = {x:i for i,x in enumerate(self.POS_51_all_tags)}
        self.POS_12_tag_to_id = {x:i for i,x in enumerate(self.POS_12_all_tags)}
        self.POS_7_tag_to_id = {x:i for i,x in enumerate(self.POS_7_all_tags)}
        self.POS_6_tag_to_id = {x:i for i,x in enumerate(self.POS_6_all_tags)}
        
        self.POS_51_id_to_tag = {i:x for i,x in enumerate(self.POS_51_all_tags)}
        self.POS_12_id_to_tag = {i:x for i,x in enumerate(self.POS_12_all_tags)}
        self.POS_7_id_to_tag = {i:x for i,x in enumerate(self.POS_7_all_tags)}
        self.POS_6_id_to_tag = {i:x for i,x in enumerate(self.POS_6_all_tags)}
        
        # setup df_display_renameer

        for x,y in zip([f"sel_{i}" for i in range(6)],self.POS_6_all_tags):
            self.df_display_renameer[x]=y
        for x,y in zip([f"maxp_{i}" for i in range(6)],self.POS_6_all_tags):
            self.df_display_renameer[x]=y
            
        # for k,v in _display_target_names.items():
        #     display_target_names[k] = f"Current Word {v}"
        #     display_target_names[k+"-"] = f"Previous Word {v}"
        #     display_target_names[k+"+"] = f"Next Word {v}"
            
        # setup df_display_replacer
        self.df_display_replacer={
            "model": self.model_to_label,
            "layer": self.layer_label_to_idx,
            "benchmark": self.display_benchmark_names,
        }
        self.display_POS_6_names = self.POS_6_id_to_tag

    def convert_df_for_display(self,df,convert_idx=False):
        
        sels=[f"sel_{i}" for i in range(6)]
        maxps=[f"maxp_{i}" for i in range(6)]
        df = df.replace(self.df_display_replacer)
        # df = df.replace({"model": self.model_to_label})
        # df = df.replace({"layer": self.layer_label_to_idx_map})
        df = df.rename(columns=self.df_display_renameer)
        # df = df.replace('_', ' ', regex=True)
        df = df.replace(r'_+', ' ', regex=True)
        df.columns = df.columns.str.replace(' ', '')
        if convert_idx:
            df = self.convert_index_for_display(df)
        return df
    
    def convert_index_for_display(self,df):
        # print(f"{len(df.index.names)=}")
        # print(f"{df.index.names=}")
        # print(f"{df.index.names[0]=}")
        if len(df.index.names) and df.index.names[0]==None:
            return df
        old_index=df.index
        new_index_dict = {x:old_index.get_level_values(i)  for i,x in enumerate(old_index.names)}
        new_index_df=pd.DataFrame.from_dict(new_index_dict)
        new_index_df=self.convert_df_for_display(new_index_df)
        new_index=pd.MultiIndex.from_frame(new_index_df)
        df = df.set_index(new_index)
        return df
    
    def df_display_values_as_percentages(self, df):
        res_df=(df*100).round(2).applymap("{}%".format).replace("0.0%","0%")
        # res_df=(df*100).round(2).applymap("{}".format).replace("0.0","0")
        return res_df
    def convert_plot_for_display(self, df):
        pass
    def prep_df_for_analysis(self, df, scale_xl_layer=True):
        cols=list(df)
        if ("layer" in cols) and (not "layer_idx" in cols):
            df = df.assign(layer_idx=df["layer"].map(self.layer_label_to_idx))
            # df["layer_idx"] = df["layer"].map(self.layer_label_to_idx)
        if ("model" in cols) and (not "model_group" in cols):
            df = df.assign(model_group=df["model"].map(self.model_to_group))
            # df["model_group"] = df["model"].map(self.model_to_group)
        if scale_xl_layer and len(df[df['layer_idx']==48]):
            mask=df["model"].isin(self.gpt2xl_models)
            df = df.assign(layer_idx=df["layer_idx"].mask(mask,df["layer_idx"]*12/48))
            # df["layer_idx"] = df["layer_idx"].mask(mask,df["layer_idx"]*12/48)
        # df = df.sort_values("model_group", key=lambda x: pd.Series.map(x,DD.group_to_groupIdx), kind="mergesort")
        df = df.sort_values("layer_idx", kind="mergesort")
        df = df.sort_values("model", key=lambda x: pd.Series.map(x,self.model_to_modelIdxs), kind="mergesort")
        return df
    def display_full(self,x):
        with pd.option_context('display.max_rows', None,
                               'display.max_columns', None,
                               'display.width', 2000,
                               # 'display.float_format', '{:20,.2f}'.format,
                               'display.max_colwidth', None):
            display(x)