import pickle
import json
from pathlib import Path
import copy
import itertools, functools
import math
import os
import numpy as np
import pandas as pd
from collections import OrderedDict

HOME=os.getenv("HOME")

try:
    # print(get_ipython())
    from tqdm.notebook import tqdm # Progress bar
    # print("ipython enabled")
except:
    from tqdm import tqdm
    # print("no ipython")


import sys
if not f"{HOME}/untrained_prior/jupyterlab/utils" in sys.path: sys.path.append(f"{HOME}/untrained_prior/jupyterlab/utils")

# ===========================================================
# ======== IPython functions
# ===========================================================
from IPython.display import display_html
def restartkernel():
    """Restart Jupyter kernel."""
    display_html("<script>Jupyter.notebook.kernel.restart()</script>", raw=True)


# ===========================================================
# ======== Print utilities
# ===========================================================
def hr(height=2):
    """Print horizontal rule."""
    display(HTML(f'<hr style="height:{height}px;">'))

def print_list(L):
    """Print list items."""
    for x in L: print(x)

def print_ilist(L):
    """Print indexed list items."""
    for i, x in enumerate(L): print(f"{i:<3}: {x}")

def print_dict(D):
    """Print dictionary."""
    print("{")
    for k, v in D.items(): print(f'  "{k}": {v},')
    print("}")

def print_kdict(D):
    """Print dictionary keys."""
    print_list(D.keys())

# Aliases
print_l = print_list
print_il = print_ilist
print_d = print_dict
print_kd = print_kdict

def unique(x): return list(dict.fromkeys(x))  # Remove duplicates while preserving order

def dFirst(dic): return dic[list(dic)[0]]  # Get first element of dict

def dOnlyFirst(dic): return {list(dic)[0]: dic[list(dic)[0]]}  # Return dict with only the first item

def first(X):  # Get first element of list or dict
    if isinstance(X, dict): return next(iter(X.values()))
    return X[0]

def get_size_of_dict(d):  # Get size of dictionary in GB
    size = sys.getsizeof(d)
    for v1 in d.values():
        size += sys.getsizeof(v1)
        if isinstance(v1, (dict, OrderedDict)): size += get_size_of_dict(v1) * 10**9
        elif isinstance(v1, (list, tuple, set)): size += sum(sys.getsizeof(x) for x in v1)
    return size / 10**9


# ===========================================================
# ========== File utilities
# ===========================================================
def lsf(d): return [x.name for x in Path(d).iterdir() if x.is_file()]  # List files in directory

def lsf_stem(d): return [x.stem for x in Path(d).iterdir() if x.is_file()]  # List filenames without extensions

def list_custom_identifiers(weight_dir=f"{HOME}/data/transformer_weights/"):
    """List custom identifiers in weight directory."""
    return [x.stem for x in Path(weight_dir).iterdir() if x.is_file()]

def get_base_model_from_identifier(x): return x.split("_")[0]  # Extract base model from identifier

def list_presaved_models(saved_model_dir=f"{HOME}/data/transformer_saved_models"):
    """List pre-saved models in saved_model directory."""
    return [x.stem for x in Path(saved_model_dir).iterdir() if x.stem != "old"]





# ===========================================================
# ========== Save and Load utilities
# ===========================================================
def save_txt(file_path, data, make_dirs=0):
    """Save text to file."""
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'w') as f: f.write(data)

def load_txt(file_path):
    """Load text from file."""
    with open(file_path, "r") as f: return f.read()

def saveb(file_path, data, make_dirs=0):
    """Save binary data with np."""
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'wb') as f: np.save(f, tuple(data))

def loadb(file_path):
    """Load binary data with np."""
    data = np.load(file_path)
    return [data[x][0] for x in data]

def load_txtl(file_path):
    """Load text lines from file."""
    with open(file_path, "r") as f: return f.readlines()

def save_dict(file_path, data, make_dirs=0):
    """Save dictionary as JSON."""
    if make_dirs: ensure_dir(file_path)
    with open(file_path, "w") as f: json.dump(data, f)

def load_dict(file_path):
    """Load dictionary from JSON."""
    with open(file_path, "r") as f: return json.load(f)

def save_pickle(file_path, obj, make_dirs=0):
    """Save object as pickle."""
    if make_dirs: ensure_dir(file_path)
    with open(file_path, 'wb') as f: pickle.dump(obj, f)

def load_pickle(file_path):
    """Load object from pickle."""
    with open(file_path, 'rb') as f: return pickle.load(f)

def ensure_dir(file_path):
    """Ensure directory exists."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory): os.makedirs(directory)

# ============================================================
# ========== Numpy utilities
# ============================================================'

def calc_np_stats(x):
    """Calculate basic stats for numpy array."""
    return [x.max(), x.min(), x.mean(), np.median(x), x.std()]

def print_np_stats(x):
    """Print basic stats for numpy array."""
    stats = calc_np_stats(x)
    stats = [f"{round(s, 5):.11}" for s in stats]
    print(", ".join([f"{k}: {v}" for k, v in zip(["max", "min", "mean", "median", "std"], stats)]))

def select_data(df, **kwargs):
    
    """Select rows in DataFrame based on column filters."""
    for k, v in kwargs.items():
        if not isinstance(v, list): v = [v]
        df = df[df[k].isin(v)]
    return df

# ============================================================
# ========== Neural network utilities
# ============================================================'

def print_layer_names(model):
    """Print layer names in model."""
    for name, _ in model.named_parameters():
        print(name)

def print_dimensions(*args, title=None, depth=0, **kwargs):
    """Print dimensions of various data structures (lists, dicts, numpy arrays, tensors, etc.)."""

    def check_if_tf_and_torch_enabled():
        """
        Check if TensorFlow and PyTorch are enabled in the environment and import them if necessary.
        Sets global flags `tf_enabled` and `torch_enabled` to indicate the status.
        """
        global tf_enabled, torch_enabled
        global tf, torch

        # Check TensorFlow
        try:
            tf_enabled  # Check if the flag exists
        except:
            tf_enabled = 'tensorflow' in list(sys.modules)  # Check if TensorFlow is imported
            if tf_enabled:
                import tensorflow as tf  # Import TensorFlow if available

        # Check PyTorch
        try:
            torch_enabled  # Check if the flag exists
        except:
            torch_enabled = 'torch' in list(sys.modules)  # Check if PyTorch is imported
            if torch_enabled:
                import torch  # Import PyTorch if available

        return

    def _print_dimensions(data, depth=0, stats=True, print_type=False, prepend=None):
        """
        Recursively print the dimensions of a given data structure.

        Args:
            data: The data to inspect.
            depth (int): Recursion depth for pretty-printing indentation.
            stats (bool): Whether to print additional statistics (e.g., mean, std) if available.
            print_type (bool): Whether to print the data type.
            prepend (str): String to prepend to the printed output (e.g., index for lists).
        """

        def format_type(data):
            """Format the data type as a string without Python class formatting."""
            return str(type(data)).replace("<class '", "").replace("'>", "")

        def print_as_list(data, start, depth=0, **kwargs):
            """Handle printing for list-like structures."""
            if len(data) > 0:
                # Print the list type and its length
                print(f"{start}{format_type(data)}({len(data)}):")
                # Recursively call _print_dimensions for each element in the list
                for i, d in enumerate(data):
                    _print_dimensions(d, depth=depth + 1, **kwargs, prepend=f"[{i}]")
            else:
                print(f"{start}{format_type(data)}({len(data)})")  # Print empty list

        def print_as_dict(data, start, depth=0, **kwargs):
            """Handle printing for dictionary-like structures."""
            dict_len = len(data.keys())
            if dict_len:
                # Print the dictionary type and length
                print(f"{start}{format_type(data)}({len(list(data))}) :=")
                # Recursively call _print_dimensions for each key-value pair
                for i, (key, value) in enumerate(data.items()):
                    if isinstance(value, str):  # Handle string values directly
                        print(f"{'    ' * depth}{{{i}}} {key}: {value}")
                    else:
                        print(f"{'    ' * depth}{{{i}}} {key}:")
                        _print_dimensions(value, depth=depth + 1, **kwargs)
            else:
                print(f"{start}{format_type(data)}({len(list(data))}):")  # Empty dictionary

        # Initial setup for indentation and printing options
        kwargs = {"stats": stats, "print_type": print_type}
        start = f"{'    ' * depth}"  # Indentation based on depth
        if prepend:
            start += f"{prepend} "  # Append prepend text (e.g., index)
        if print_type:
            start += f"(type={format_type(data)}), "

        # Case 1: Handle strings
        if isinstance(data, str):
            print(f"{start}{format_type(data)}='{data}'")

        # Case 2: Handle basic types (int, float, bool, etc.)
        elif isinstance(data, (int, float, complex, bool, bytes)) or data is None:
            print(f"{start}{format_type(data)}={data}")

        # Case 3: Handle dictionaries (including OrderedDict)
        elif isinstance(data, (dict, OrderedDict)):
            print_as_dict(data, start, depth=depth, **kwargs)

        # Case 4: Handle list-like structures (list, tuple, set, etc.)
        elif isinstance(data, (set, list, tuple, frozenset)):
            print_as_list(data, start, depth=depth, **kwargs)

        # Case 5: Handle PyTorch tensors
        elif torch_enabled and isinstance(data, torch.Tensor):
            if data.dim() == 0:
                # Scalar tensor
                print(f"{start}dims = {data.dim()}, dtype = {data.dtype}")
            elif data.dtype == torch.bool:
                # Boolean tensor
                print(f"{start}size = {tuple(data.size())}, dtype = {data.dtype}")
            else:
                if stats:
                    # Calculate mean and std for non-boolean tensors
                    std, mean = torch.std_mean(data.detach().clone().float())
                    print(f"{start}shape={tuple(data.size())}, mean = {mean}, std = {std}")
                else:
                    print(f"{start}shape={tuple(data.size())}")

        # Case 6: Handle TensorFlow tensors
        elif tf_enabled and isinstance(data, tf.Tensor):
            if len(data) > 0:
                # Tensor with data
                if stats:
                    # Print mean and std if available
                    print(f"{start}shape={tuple(data.shape)}, mean = {tf.reduce_mean(data)}, std = {tf.math.reduce_std(data)}")
                else:
                    print(f"{start}shape={tuple(data.shape)}")
            else:
                # Empty tensor
                print(f"{start}len={len(data)}")

        # Case 7: Handle numpy arrays
        elif isinstance(data, np.ndarray):
            dtype = data.dtype
            if len(data) > 0:
                # Non-empty array
                if stats and not "<U" in str(dtype):  # Skip string arrays for stats
                    print(f"{start}np.ndarray({data.shape}), dtype={dtype}, mean = {data.mean()}, std = {data.std()}")
                else:
                    print(f"{start}np.ndarray({data.shape}), dtype={dtype}")
            else:
                # Empty array
                print(f"{start}np.ndarray({len(data)}), dtype={dtype}")

        # Case 8: Handle unknown data types
        else:
            start = f"{'    ' * depth}(unknown type={format_type(data)}), "
            # Attempt to print as a dictionary
            try:
                print_as_dict(data, start, depth=depth, **kwargs)
                return
            except:
                pass
            # Attempt to print as a list
            try:
                print_as_list(data, start, depth=depth, **kwargs)
                return
            except:
                pass
            # As a last resort, just print the data
            print(f"{start}data='{data}'")
        return

    # Check if TensorFlow and PyTorch are available
    check_if_tf_and_torch_enabled()

    # If no data provided, just print an empty line
    if not len(args):
        print()

    # If a title is provided, print it with a separator
    if title:
        print("=" * 100)
        print(title)
        for data in args:
            _print_dimensions(data, depth=depth + 1, **kwargs)
        print("=" * 100)
    else:
        # Print dimensions for each argument
        for data in args:
            _print_dimensions(data, depth=depth, **kwargs)


# ===========================================================
# ========== Debugging utilities
# ===========================================================

import inspect
import re
def debugPrint(x, out_len=1000, end="\n"):
    """Debug print a variable name and value."""
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    x = eval(r)
    full_statement = f"*** {r} (type={type(x)}): = {repr(x)}"
    if len(full_statement) > out_len:
        full_statement = full_statement[:out_len - 6].rstrip(" .,") + " ..."
    if len(full_statement) > 300 and end == "\n": end = "\n\n"
    print(full_statement, end=end)

