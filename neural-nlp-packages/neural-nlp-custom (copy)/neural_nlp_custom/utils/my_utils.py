import pickle
import json
from pathlib import Path
import copy
import itertools, functools
import math

import os
from collections import OrderedDict
import numpy as np
import pandas as pd

try:
    # print(get_ipython())
    from tqdm.notebook import tqdm
    # print("ipython enabled")
except:
    from tqdm import tqdm
    # print("no ipython")


import sys
if not f"{HOME}/code/jupyterlab/utils" in sys.path: sys.path.append(f"{HOME}/code/jupyterlab/utils")
# from my_utils import *
# from my_latex_tools import *
# from ProjectManager import ProjectManager
# PM = ProjectManager()


# ====================================
# ========== IPython functions
# ====================================
from IPython.display import display_html
def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)
    
# try:
#     print(get_ipython())
#     print("ipython enabled")
# except:
#     print("no ipython")
#     pass

# ===========================================================
# ======== print utils
# ===========================================================

def hr(height=2): 
    display(HTML('<hr style="height:{}px; border-width:0; color:black; background-color:black;margin:0.1em;">'.format(height)))
def print_list(L):
    for x in L: print(x)
def print_ilist(L):
    for i,x in enumerate(L): print("{:<3}: {}".format(i,x))
def print_dict(D):
    print("{")
    for k,v in D.items(): 
        print('  "{}": {},'.format(k,v))
    print("}")
def print_kdict(D):print_list(D.keys())
    
print_l=print_list
print_il=print_ilist
print_d=print_dict
print_kd=print_kdict

def unique(x): return list(dict.fromkeys(x))
def dFirst(dic):
    return dic[list(dic)[0]]
def dOnlyFirst(dic):
    return {list(dic)[0]:dic[list(dic)[0]]}

def get_size_of_dict(d):
    size=sys.getsizeof(d)
    for k1,v1 in d.items():
        size+=sys.getsizeof(v1)
        if type(v1) in [dict,OrderedDict]:
            size+=get_size_of_dict(v1)*(10**9)
            # for k2,v2 in v1.items():
                # size+=sys.getsizeof(v2)
        elif type(v1) in [list,tuple,set]:
            for x in v1:
                size+=sys.getsizeof(x)
    return size/10**9
# ===========================================================
# ========== getting text from files
# ===========================================================
from pathlib import Path
def lsf(d): return [x.name for x in Path(d).iterdir() if x.is_file()]
def lsf_stem(d): return [x.stem for x in Path(d).iterdir() if x.is_file()]

# for get_activations
from pathlib import Path
def list_custom_identifiers(weight_dir=f"{HOME}/data/transformer_weights/"): return [x.stem for x in Path(weight_dir).iterdir() if x.is_file()]
def get_base_model_from_identifier(x): return x.split("_")[0]
def list_presaved_models(saved_model_dir=f"{HOME}/data/transformer_saved_models"):
    return [x.stem for x in Path(saved_model_dir).iterdir() if not x.stem=="old"]




# ===========================================================
# Save utils
from pathlib import Path
def save_txt(file_path,data,make_dirs=0):
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'w') as f:
        f.write(data)
def load_txt(file_path):
    with open(file_path,"r")as f:
        d = f.read()
    return d
def saveb(file_path,data,make_dirs=0): # save with np
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'wb') as f:
        np.save(f, tuple(data))
def loadb(file_path): # save with np
    data = np.load(file_path)
    data= [data[x][0] for x in data]
    # print(data)
    return data
def load_txtl(file_path):
    with open(file_path,"r")as f:
        d = f.readlines()
    return d
def save_dict(file_path,data,make_dirs=0):
    if make_dirs: ensure_dir(file_path)
    with open(file_path, "w") as f:
         json.dump(data,f)
def load_dict(file_path):
    with open(file_path, "r") as f:
         data = json.load(f)
    return data
def save_dictb(file_path,data,make_dirs=0):
    if make_dirs: ensure_dir(file_path)
    with open(file_path, "wb") as f:
         json.dump(data,f)
def load_dictb(file_path):
    with open(file_path, "rb") as f:
         data = json.load(f)
    return data
def save_pickle(file_path, obj,make_dirs=0):
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        D = pickle.load(f)
    return D
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
# ============================================================
# ========== Numpy tools
# ============================================================'

def calc_np_stats(x):
    return [x.max(),x.min(),x.mean(),np.median(x),x.std()]
def print_np_stats(x):
    outs = calc_np_stats(x)
    outs = [f"{str(round(s,5)):.11}" for s in outs]
    outs = [f"{str(s): <12}" for s in outs]
    print(", ".join([f"{k}: {v}" for k,v in zip(["max","min","mean","median","std"],outs)]))
def print_np_stats2(x):
    outs = x
    outs = [f"{str(round(s,5)):.11}" for s in outs]
    outs = [f"{str(s): <12}" for s in outs]
    print(", ".join([f"{k}: {v}" for k,v in zip(["max","min","mean","median","std"],outs)]))

# def is_np_array_standardized(x):
#     return (abs(np.mean(x))<0.01) and (abs(np.std(x)-1)<0.01)
# def try_to_standardize_dict_of_dict_of_nparrays_in_place(x):
#     for k1,v2 in x.items():
#         for k2,v2 in v1.items():
#             try:
#                 if is_np_array_standardized(v2):
#                     # print(f"array already standardized with key={k}")
#                     pass
#                 else:
#                     x[k1][k2]= standardize_np_array(v2)
#                     # print(f"array standardized with key={k}")
#             except:
#                 # print(f"array failed to standardize with key={k}")
#                 pass
# def try_to_standardize_dict_of_nparrays_in_place(x):
#     for k,v in x.items():
#         try:
#             if is_np_array_standardized(v):
#                 print(f"array already standardized with key={k}")
#             else:
#                 x[k]= standardize_np_array(v)
#                 print(f"array standardized with key={k}")
#         except:
#             print(f"array failed to standardize with key={k}")
# def try_to_standardize_np_array(x):
#     try:
#         return standardize_np_array(x)
#     except:
#         return x
# def standardize_np_array(x):
#     if is_np_array_standardized(x): 
#         # already standardized
#         return x
#     else:
#         # standardize 
#         return (x - np.mean(x)) / np.std(x)

def select_data(df,**kwargs):
    for k,v in kwargs.items():
        if not type(v)==list: v=[v]
        df = df[df[k].isin(v)]
    return df

# ============================================================
# ========== Nerural network utils:
# ==========    print_layer_names, print_dimensions (print_b)
# ============================================================
def print_layer_names(model): # print_layer_names(model)
    """
    Prints the names of all layers in a Transformers GPT2Model.
    """
    for name, _ in model.named_parameters():
        print(name)

def print_dimensions(*args, title=None, depth=0, **kwargs):
    def check_if_tf_and_torch_enabled():
        global tf_enabled, torch_enabled
        global tf, torch
        try:
            tf_enabled
        except:
            tf_enabled='tensorflow' in list(sys.modules)
            res="enabled" if tf_enabled else "disabled"
            # print(f"ben_utils support for TF is {res}.")
            if tf_enabled:
                import tensorflow as tf
        try:
            torch_enabled
        except:
            torch_enabled='torch' in list(sys.modules)
            res="enabled" if torch_enabled else "disabled"
            # print(f"ben_utils support for torch is {res}.")
            if torch_enabled:
                import torch
        # return tf_enabled, torch_enabled
        return
    def _print_dimensions(data, depth=0, stats=True, print_type=False, prepend=None):

        def format_type(data):
            return str(type(data)).replace("<class '","").replace("'>","")

        def print_as_list(data, start, depth=0, **kwargs):
            if len(data) > 0:
                print(f"{start}{format_type(data)}({len(data)}):")
                for i, d in enumerate(data):
                    # print(i,end="")
                    _print_dimensions(d, depth=depth+1, **kwargs, prepend=f"[{i}]")  # recursively call with increased depth
            else:
                print(f"{start}{format_type(data)}({len(data)})")

        def print_as_dict(data, start, depth=0, **kwargs):
            dict_len=len(data.keys())
            if dict_len:
                print(f"{start}{format_type(data)}({len(list(data))}) :=")
                for i, (key, value) in enumerate(data.items()):
                    # print(i,end="")
                    if isinstance(value, str):
                        print(f"   {'    ' * depth}" + "{" + str(i) + "}" + f" {key}: {value}")
                    else:
                        print(f"   {'    ' * depth}" + "{" + str(i) + "}" + f" {key}:")
                        _print_dimensions(value, depth=depth+1, **kwargs)  # recursively call with increased depth
            else:
                print(f"{start}{format_type(data)}({len(list(data))}):")
        kwargs={"stats":stats,"print_type":print_type}
        start=f"{'    ' * depth}" # calculate indent based on recursion depth
        if prepend:
            start+=f"{prepend} "
        if print_type:
            start+=f"(type={format_type(data)}), "

        # strings
        if any((isinstance(data, x) for x in [str])):
            print(f"{start}{format_type(data)}='{data}'")

        # strings-like
        elif any((isinstance(data, x) for x in [int,float,complex,bool,bytes])) or data is None:
            print(f"{start}{format_type(data)}={data}")

        # dicts
        elif any((isinstance(data, x) for x in [dict,OrderedDict])):
            print_as_dict(data, start, depth=depth, **kwargs)

        # Lists
        elif any((isinstance(data, x) for x in [set, list, tuple,frozenset])):
            print_as_list(data, start, depth=depth, **kwargs)

        # torch tensor
        elif torch_enabled and isinstance(data, torch.Tensor):
            if data.dim() == 0:
                print(f"{start}dims = {data.dim()}, dtype = {data.dtype}")
            elif data.dtype == torch.BoolTensor:
                print(f"{start}size = {tuple(data.size())}, dtype = {data.dtype}")
            else:
                if stats:
                    std, mean=torch.std_mean(data.detach().clone().type(torch.FloatTensor))
                    print(f"{start}shape={tuple(data.size())}, mean = {mean},  = {std}")
                else:
                    print(f"{start}shape={tuple(data.size())}")

        # tensorflow tensor
        elif tf_enabled and isinstance(data, tf.Tensor):
            if len(data) > 0:
                if stats:
                    print(f"{start}shape={tuple(data.shape)}, mean = {tf.reduce_mean(data)}, std = {tf.math.reduce_std(data)}")
                else:
                    print(f"{start}shape={tuple(data.shape)}")
            else:
                print(f"{start}len={len(data)}")

        # Numpy array
        elif isinstance(data, np.ndarray):
            dtype=data.dtype
            if len(data) > 0:
                if stats and not "<U" in str(dtype):
                    print(f"{start}np.ndarray({data.shape}), dtype={dtype}, mean = {data.mean()}, std = {data.std()}")
                else:
                    print(f"{start}np.ndarray({data.shape}), dtype={dtype}")
            else:
                print(f"{start}np.ndarray({len(data)}), dtype={dtype}")

        # otherwise, guess
        else:  
            start=f"{'    ' * depth}(unknown type={format_type(data)}), "
            # try printing as a dict
            try:
                print_as_dict(data, start, depth=depth, **kwargs)
                return
            except:
                pass

            # try printing as a list
            try:
                print_as_list(data, start, depth=depth, **kwargs)
                return
            except:
                pass

            # as last resort, just hard print whatever it is
            print(f"{start}data='{data}'")
        return
    check_if_tf_and_torch_enabled()
    if not len(args):
        print()
    if title:
        print("="*100)
        print(title)
        for data in args:
            _print_dimensions(data, depth=depth+1, **kwargs)
        print("="*100)
    else:
        for data in args:
            _print_dimensions(data, depth=depth, **kwargs)

print_b=print_dimensions


# ===========================================================
# ========== function for debugging calls
# ===========================================================
import inspect
import re
def debugPrint(x, out_len=1000, end="\n"):
    # base from: https://stackoverflow.com/questions/32000934/print-a-variables-name-and-value
    # alt: print(f"{ys=}")
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    x = eval(r)
    # full_statement=f"Debug: {r} (type={type(x)}): = {repr(x)}"
    full_statement=f"*** {r} (type={type(x)}): = {repr(x)}"
    if len(full_statement)>out_len:
        # full_statement = f"{full_statement< out_len}"+ " " * (len(full_statement)-3) + "   ..."
        # full_statement = f"{full_statement: <{out_len}} ..."
        full_statement = full_statement[:out_len-6].rstrip(" .,") + " ..."
    if len(full_statement)>300 and end=="\n": end="\n\n"
    print(full_statement,end=end)
# debugPrint(3*4)





# ====================================
# ========== Notes and tricks functions
# ====================================

# # from tqdm import tqdm
# # use this instead:
# from tqdm.notebook import tqdm

# # import itertools
# # https://docs.python.org/3/library/itertools.html
# itertools.chain.from_iterable(['ABC', 'DEF']) --> A B C D E F