# ============================================================
# ========== print_dimensions and print_layer_names functions
# ============================================================
# import numpy as np
# import torch
# import tensorflow as tf
# from collections import OrderedDict
# from tqdm.notebook import tqdm

# ====================================
# ========== IPython functions
# ====================================

# from tqdm import tqdm
# use this instead:
try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm


# import itertools
# https://docs.python.org/3/library/itertools.html
itertools.chain.from_iterable(['ABC', 'DEF']) # --> A B C D E F


# ====================================
# ========== IPython functions
# ====================================

# test if run in an ipython environment
try:
    print(get_ipython()) # error if not
    print("ipython enabled")
except:
    print("no ipython")
    pass

from IPython.display import display_html
def restartkernel() :
    display_html("<script>Jupyter.notebook.kernel.restart()</script>",raw=True)

    
# ====================================
# ========== Notes and tricks functions
# ====================================

from contextlib import suppress
with suppress(NameError): del Xss; del ys