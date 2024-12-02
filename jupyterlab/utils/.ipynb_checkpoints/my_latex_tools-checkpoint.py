from pathlib import Path
import matplotlib
# import seaborn as sb
import pandas as pd
import sys
import os


from my_utils import *

HOME=os.getenv("HOME")
tbl_save_locaiton = f"{HOME}/data/thesis_tex/tbl"
fig_save_locaiton  = f"{HOME}/data/thesis_tex/fig"
fimg_save_locaiton = f"{HOME}/data/thesis_tex/fig"

def replace_all(text, dic):
    """Replace multiple substrings in a string based on a dictionary mapping."""
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

def save_df_as_tex(df, fn='test_df', v=0, index=False, **kwargs):
    """
    Save a DataFrame as a LaTeX table with a reference for inclusion in a LaTeX document.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        fn (str, optional): Filename for saving the LaTeX table (default is 'test_df').
        v (int, optional): Verbosity level, 1 for additional output (default is 0).
        index (bool, optional): Whether to include DataFrame indices in the LaTeX table (default is False).
        **kwargs: Additional arguments to pass to `to_latex()`.

    Prints:
        The LaTeX code for including the table in a LaTeX document.
    """
    assert type(df) == pd.core.frame.DataFrame, f"error, save_df_as_tex unable to save image type(df) = {str(type(df))}"

    # Generate filenames and labels
    fnt=f"{fn}"
    label=f"tbl:{fn}"
    raw_data_table=df.to_latex(index=index, **kwargs)
    clabel=label.replace("_"," ")

    # LaTeX template for the table
    front_end_tex_template = "\n    ".join([x.strip() for x in r"""
        \begin{table}
        \centering
        \input{../inputs/tbl/__FNT__}
        \caption{caption for __CLABLE__}
        \label{__LABLE__}
        \end{table} % \ref{__LABLE__}
        """.strip().split("\n")])

    # Replace placeholders in template
    front_end_tex = front_end_tex_template.replace("__LABLE__", label).replace("__FNT__", fnt).replace("__CLABLE__", clabel)
    raw_data_tex = f"{raw_data_table}\n".replace(r"\begin{table}", "").replace(r"\end{table}", "")

    # Save the LaTeX file
    save_txt(Path(tbl_save_locaiton, f"{fnt}.tex"), raw_data_tex)

    # Print the LaTeX template for inclusion
    print(front_end_tex)
    if v:
        print("="*25)
        print(raw_data_tex)
    print("="*100)


# print(fig_tex_template)
def save_figure_for_thesis(img, fn='test_img', label="", use_svg=True, v=0):
    """
    Save a figure in SVG or PDF format for inclusion in a LaTeX document.

    Args:
        img (matplotlib.figure.Figure): The figure to save.
        fn (str, optional): Filename for saving the figure (default is 'test_img').
        label (str, optional): LaTeX label for referencing the figure (default is "").
        use_svg (bool, optional): Whether to save in SVG format (default is True). If False, saves in PDF format.
        v (int, optional): Verbosity level, 1 for additional output (default is 0).

    Prints:
        The LaTeX code for including the figure in a LaTeX document.
    """
    if str(type(img)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>":
        img = img.figure
    assert type(img) == matplotlib.figure.Figure, "error, save_figure_for_thesis unable to save figure type(img) = " +str(type(img))

    # LaTeX template for the figure
    front_end_tex_template = "\n    ".join([x.strip() for x in r"""
        \begin{figure}
        \centering
        \includesvg[width=\linewidth]{inputs/fig/__FNP__}
        \caption{caption for __CLABLE__}
        \label{__LABLE__}
        \end{figure} % \ref{__LABLE__}
        """.strip().split("\n")])

    # Replace placeholders in template
    fn = fn.lower()
    label = f"fig:{fn}"
    clabel = label.replace("_", " ")
    replace_dict = {
        "__LABLE__": label,
        "__CLABLE__": clabel,
        "__FNP__": fn,
    }
    if not use_svg:
        replace_dict[r"\includesvg"] = r"\includegraphics"
    front_end_tex = replace_all(front_end_tex_template, replace_dict)

    # Save the figure as SVG or PDF
    if use_svg:
        img.savefig(Path(fimg_save_locaiton, f"{fn}.svg"), format='svg', bbox_inches='tight')
    else:
        img.savefig(Path(fimg_save_locaiton, f"{fn}.pdf"), format='pdf', dpi=1200, bbox_inches='tight')

    # Print the LaTeX template for inclusion
    print(front_end_tex)
    print("="*100)



################################################################
# LaTeX accents replacement
# https://stackoverflow.com/questions/4578912/replace-all-accented-characters-by-their-latex-equivalent
def format_symbols_to_tex(text):
    """
    Replace accented characters with their LaTeX equivalents.

    Args:
        text (str): Input text containing accented characters.

    Returns:
        str: The text with accented characters replaced by LaTeX commands.
    """
    latexAccents = {
        u"à": "\\`a ", u"è": "\\`e ", u"ì": "\\`\\i ", u"ò": "\\`o ", u"ù": "\\`u ", u"ỳ": "\\`y ",
        u"À": "\\`A ", u"È": "\\`E ", u"Ì": "\\`\\I ", u"Ò": "\\`O ", u"Ù": "\\`U ", u"Ỳ": "\\`Y ",
        u"á": "\\'a ", u"é": "\\'e ", u"í": "\\'\\i ", u"ó": "\\'o ", u"ú": "\\'u ", u"ý": "\\'y ",
        u"Á": "\\'A ", u"É": "\\'E ", u"Í": "\\'\\I ", u"Ó": "\\'O ", u"Ú": "\\'U ", u"Ý": "\\'Y ",
        u"â": "\\^a ", u"ê": "\\^e ", u"î": "\\^\\i ", u"ô": "\\^o ", u"û": "\\^u ", u"ŷ": "\\^y ",
        u"Â": "\\^A ", u"Ê": "\\^E ", u"Î": "\\^\\I ", u"Ô": "\\^O ", u"Û": "\\^U ", u"Ŷ": "\\^Y ",
        u"ä": "\\\"a ", u"ë": "\\\"e ", u"ï": "\\\"\\i ", u"ö": "\\\"o ", u"ü": "\\\"u ", u"ÿ": "\\\"y ",
        u"Ä": "\\\"A ", u"Ë": "\\\"E ", u"Ï": "\\\"\\I ", u"Ö": "\\\"O ", u"Ü": "\\\"U ", u"Ÿ": "\\\"Y ",
        u"ç": "\\cc ", u"Ç": "\\cC ", u"œ": "\\oe ", u"Œ": "\\OE ", u"æ": "\\ae ", u"Æ": "\\AE ",
        u"å": "\\aa ", u"Å": "\\AA ", u"ø": "\\o ", u"Ø": "\\O ", u"ß": "\\ss ", u"¡": "!`", u"¿": "?`",
    }
    for x, y in latexAccents.items():
        if x in text:
            text = text.replace(x, y)
    return text


# lazy importing functions, since this was coded for a different project


# utils
# path utils
import os, sys
import itertools, functools
import pickle
import math
import json
import numpy as np
from pathlib import Path
import copy


# Misc
def to_unix_list(l): return "\n".join([str(x) for x in l])# for x in list(map(list, zip(*[*argv])))])
def neglogprob_to_surprisal(x): return x/math.log10(2)
def softmax_funtion(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
def round_to_n(x, sig): 
    try:
        res = round(x, sig-int(math.floor(math.log10(abs(x))))-1)
    except:
        res = x
    return res

# iter utilities (from Lua)
def i_iter():
    i=0
    while 1: yield i; i+=1
def pairs(*argv):
    it=[iter(collection) for collection in argv]
    for i in range(0,len(argv[0])):
        yield (x.__next__() for x in it)
def ipairs(*argv):
    it=[i_iter()]+[iter(collection) for collection in argv]
    for i in range(0,len(argv[0])):
        yield (x.__next__() for x in it)
def transpose(l): return list(map(list, zip(*l)))

# alignment util
def to_str(x): return [_to(y,str) for y in x]
def to_int(x): return [_to(y,int) for y in x]
def to_float(x): return [_to(y,float) for y in x]
def _to(x,fun): 
    try: 
        if np.isnan(x): return None
    except: pass
    return fun(x)

# from: https://stackoverflow.com/questions/579310/formatting-long-numbers-as-strings
def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
