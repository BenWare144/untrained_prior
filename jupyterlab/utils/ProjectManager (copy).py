import os
HOME=os.getenv("HOME")
import sys
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd

try:
    from tqdm.notebook import tqdm
except:
    from tqdm import tqdm

# Local imports
from my_utils import *


### Example usage:
# PM = ProjectManager()
# data = PM.fetch_data_by_label("activations", "double_2")

### saving
# PM.save_dataset("word_features", PM.fn_word_frequencies, data=word_frequencies)
# PM.get_files("transformer_weights")

### loading
# sentence_features = PM.load_dataset("sentence_features")
# sentence_features
# word_features = PM.load_dataset("word_features")
# word_features
# gpt_input = PM.load_dataset("gpt_input")
# gpt_input

class ProjectManager:
    """
    Manages datasets, model activations, and project-specific file handling for language model experiments.

    This class provides methods to load, save, and process data for various models, 
    including activations, word features, and sentence features. It handles caching 
    and offers tools to standardize and prepare data for analysis.

    Attributes:
    -----------
    Xss : dict
        Stores activations for all models.
    ys : dict
        Stores target values for all classification and regression tasks.
    directories : dict
        Contains paths to various project directories (e.g., transformer weights, activations, scores).
    model_label_map : dict
        Maps model identifiers to display labels for easier interpretation.
    layers_dict : dict
        Maps model names to their respective layers.
    """
    Xss={}
    ys={}
    ys_6={}
    Xss_6={}
    is_in_POS_6=[]
    base_dir = Path(f"{HOME}/data")
    directories = {
        "transformer_weights": base_dir / "transformer_weights",
        "word_features": base_dir / "word_features",
        "sentence_features": base_dir / "sentence_features",
        "scores": base_dir / "scores",
        "activations": base_dir / "activations",
        "context_lengths": base_dir / "context_lengths",
        "analysis_res": base_dir / "analysis_res",
        "thesis_tex": base_dir / "thesis_tex",
        "tbl": base_dir / "thesis_tex" / "tbl",
        "fig": base_dir / "thesis_tex" / "fig",
        "fimg": base_dir / "thesis_tex" / "fimg",
        "hf_files": base_dir / "hf_files",
        "gpt_input": base_dir / "gpt_input",
        "exp1": base_dir / "experiment_1_results",
        "exp2": base_dir / "experiment_2_results",
        "exp3": base_dir / "experiment_3_results",
        "exp3_test": base_dir / "experiment_3_test",
    }
    fn_base_sent_features = "base_sentence_features.csv"
    fn_base_word_features = "base_word_features.csv"
    fn_word_indexes = "word_indexes.csv"
    fn_pos_features = "pos_features.csv"
    # fn_pos_names = "pos_names.csv"
    fn_word_frequencies = "word_frequencies.csv"
    fn_tree_depth = "tree_depth.csv"
    fn_function = "function.csv"
    
    all_models = (
        ['gpt2-xl','gpt2-xl-untrained_1','gpt2']+
        [f"gpt2-untrained_{i}" for i in range(1,10)]+
        [f"gpt2-untrained_{i}_weight_config_all" for i in range(1,10)]
    )
    gpt2xl_models=all_models[:2]
    gpt2_models=all_models[2:]
    # Mapping between model labels and their display labels
    layers_gpt2xl=["drop"]+[f'encoder.h.{i}' for i in range(48)]
    layers_gpt2=layers_gpt2xl[:13]
    layers_dict={}
    layers_to_idx={}
    model_label_map = {
        'gpt2-xl': 'XL-Trained',
        'gpt2-xl-untrained_1': 'XL-Untrained_1',
        'gpt2': 'Trained',
        'gpt2-untrained_1_weight_config_all': 'Guassian_1',
        'gpt2-untrained_2_weight_config_all': 'Guassian_2',
        'gpt2-untrained_3_weight_config_all': 'Guassian_3',
        'gpt2-untrained_4_weight_config_all': 'Guassian_4',
        'gpt2-untrained_5_weight_config_all': 'Guassian_5',
        'gpt2-untrained_6_weight_config_all': 'Guassian_6',
        'gpt2-untrained_7_weight_config_all': 'Guassian_7',
        'gpt2-untrained_8_weight_config_all': 'Guassian_8',
        'gpt2-untrained_9_weight_config_all': 'Guassian_9',
        'gpt2-untrained_1': 'Untrained_1',
        'gpt2-untrained_2': 'Untrained_2',
        'gpt2-untrained_3': 'Untrained_3',
        'gpt2-untrained_4': 'Untrained_4',
        'gpt2-untrained_5': 'Untrained_5',
        'gpt2-untrained_6': 'Untrained_6',
        'gpt2-untrained_7': 'Untrained_7',
        'gpt2-untrained_8': 'Untrained_8',
        'gpt2-untrained_9': 'Untrained_9',}
    
    # Define ordered groups of models
    model_group_map={'gpt2-xl': 'XL-Trained',
        'gpt2-xl-untrained_1': 'XL-Untrained',
        'gpt2': 'Trained',
        'gpt2-untrained_1': 'Untrained',
        'gpt2-untrained_2': 'Untrained',
        'gpt2-untrained_3': 'Untrained',
        'gpt2-untrained_4': 'Untrained',
        'gpt2-untrained_5': 'Untrained',
        'gpt2-untrained_6': 'Untrained',
        'gpt2-untrained_7': 'Untrained',
        'gpt2-untrained_8': 'Untrained',
        'gpt2-untrained_9': 'Untrained',
        'gpt2-untrained_1_weight_config_all': 'Guassian',
        'gpt2-untrained_2_weight_config_all': 'Guassian',
        'gpt2-untrained_3_weight_config_all': 'Guassian',
        'gpt2-untrained_4_weight_config_all': 'Guassian',
        'gpt2-untrained_5_weight_config_all': 'Guassian',
        'gpt2-untrained_6_weight_config_all': 'Guassian',
        'gpt2-untrained_7_weight_config_all': 'Guassian',
        'gpt2-untrained_8_weight_config_all': 'Guassian',
        'gpt2-untrained_9_weight_config_all': 'Guassian',}
    model_groups = {
        "main": ["gpt2"]+[f"gpt2-untrained_{i}" for i in range(10)]+["all"],
        "single": [f"single_{i}" for i in range(12)],
        "doubles": [f"double_{i}" for i in range(6)],
        "quads":['attns','mlps','lns'],
    }

    # Part of Speech tags
    # pos documentation: https://data.mendeley.com/datasets/zmycy7t9h9/2
    # where XX is for pos tag missing, and -LRB-/-RRB- is "(" / ")".
    POS_51_all_tags=["XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
           "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
           "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", "WDT", "WP", "WP$", "WRB"] 
    POS_12_all_tags=["VERB","NOUN","PRON","ADJ","ADV","ADP","CONJ","DET","NUM","PRT","X","."]
    POS_7_all_tags=["Noun","Verb","Adposition","Determiner","Adjective","Adverb","X"]
    POS_6_all_tags=["Noun","Verb","Adposition","Determiner","Adjective","Adverb"]
    POS_12_to_POS_7={
        "NOUN":"Noun",
        "PRON":"Noun",
        "VERB":"Verb",
        "ADP":"Adposition",
        "DET":"Determiner",
        "ADJ":"Adjective",
        "ADV":"Adverb",
        "CONJ":"X",
        "NUM":"X",
        "PRT":"X",
        "X":"X",
        ".":"X"}
    POS_51_tag_to_id={x:i for i,x in enumerate(POS_51_all_tags)}
    POS_12_tag_to_id={x:i for i,x in enumerate(POS_12_all_tags)}
    POS_7_tag_to_id={x:i for i,x in enumerate(POS_7_all_tags)}
    POS_6_tag_to_id=POS_7_tag_to_id

    pos_names=["XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
           "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
           "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", "WDT", "WP", "WP$", "WRB",]
    sample_idx_min=386
    sample_idx_max=8344
    classification_labels = []
    for x in ['words','word_idx','predicate_lemmas', 'predicate_framenet_ids', 'word_senses', 'named_entities', 'function', 'tree_depth']:
        classification_labels+=[x, x + "-", x + "+"]
    my_classification_targets = []
    for x in ['function', 'tree_depth','word_idx']:
        my_classification_targets+=[x, x + "-", x + "+"]
    reggression_labels = [] # reggression_labels=['sentence_idx','unigram_probs', 'unigram_probs-', 'unigram_probs+', 'bigram_probs', 'bigram_probs-', 'bigram_probs+', 'trigram_probs','trigram_probs-', 'trigram_probs+']
    for x in ['sentence_idx','unigram_probs', 'bigram_probs', 'trigram_probs']:
        reggression_labels+=[x, x + "-", x + "+"]
    pos_labels = [] # pos_labels=['pos_tags','pos_tags-', 'pos_tags+']
    for x in ['pos_tags', 'POS_12_id', 'POS_7_id']:
        pos_labels+=[x, x + "-", x + "+"]

    sels=[f"sel_{i}" for i in range(6)]
    maxps=[f"maxp_{i}" for i in range(6)]
    
    def __init__(self):
        self.file_cache = {}
        self.layers_dict={
            **{x:self.layers_gpt2xl for x in self.gpt2xl_models},
            **{x:self.layers_gpt2 for x in self.gpt2_models}
            }
        self.layers_to_idx={x:i for i,x in enumerate(self.layers_gpt2xl)}
        
    def _get_model_label(self, display_label):
        for model_label, disp_label in self.model_label_map.items():
            if disp_label == display_label or model_label == display_label:
                return model_label
        raise ValueError(f"Unknown display label: {display_label}")
    def _get_display_label(self, model_label):
        for _model_label, display_label in self.model_label_map.items():
            if model_label == _model_label or model_label == display_label:
                return display_label
        raise ValueError(f"Unknown display label: {display_label}")

    def get_base_and_full_labels(self, model_label):
        if model_label in ["gpt2","gpt2-untrained", "gpt2-xl","gpt2-xl-untrained"]:
            base_label=model_label
            full_label=model_label
        elif "_weight_config_" in model_label:
            base_label="gpt2-untrained"
            full_label = f"{model_label}"
        else:
            base_label="gpt2-untrained"
            full_label = f"{base_label}_weight_config_{model_label}"
        return base_label, full_label

    def check_for_existing_patterns(self, dir_key):
        return [x for x in self.all_models if self.do_patterns_exist(dir_key, x)]

    def do_patterns_exist(self, dir_key, model_label):
        files=self.generate_file_patterns(dir_key, model_label)
        # print(dir_key, model_label,files)
        if isinstance(files,tuple):
            files=list(files)
        else:
            files=[files]
        return all(Path(self.directories[dir_key],f).is_file() for f in files)

    def generate_file_patterns(self, dir_key, model_label):
        """
        Generate file patterns for a given model label and directory key.
        """
        base_label, full_label = self.get_base_and_full_labels(model_label)
        if dir_key == "activations":
            return (
                f"{full_label}__activations_v1_i0.npy",
                f"{full_label}__activations_v1_i0_metadata.csv",
                f"{full_label}_layers.txt"
            )
        elif dir_key == "scores":
            # The scores seem to have a timestamp, so we use a wildcard for that part
            return f"{full_label}_*score_raw.csv"
        elif dir_key == "word_features" or dir_key == "sentence_features":
            # Assuming a generic pattern for word and sentence features; adjust as needed
            return f"{full_label}_*.csv"
        elif dir_key == "transformer_saved_models":
            return f"E1_Schrimpfs_{base_label}"
        elif dir_key == "transformer_weights":
            return f"{full_label}.pt"
        # Add more dir_key conditions as needed
        else:
            raise ValueError(f"Unsupported directory key for pattern generation: {dir_key}")

    def fetch_data_by_label(self, dir_key, display_label):
        model_label = self._get_model_label(display_label)
        base_label, full_label = self.get_base_and_full_labels(model_label)

        if dir_key == "activations":
            pattern = f"{full_label}__activations*.*"
        elif dir_key == "scores":
            pattern = f"{full_label}_*score_raw.csv"
        else:
            raise ValueError(f"Unsupported directory key for fetching: {dir_key}")

        files = self.get_files(dir_key, pattern)

        if dir_key == "activations":
            data = [np.load(file) for file in files if file.suffix == ".npy"]
        elif dir_key == "scores":
            data = [pd.read_csv(file) for file in files]
        else:
            data = []
        return data

    def get_files(self, dir_key, pattern="*"):
        """
        Get all files in a directory matching a pattern.
        """
        dir_path = self.directories.get(dir_key)
        if not dir_path:
            raise ValueError(f"Unknown directory key: {dir_key}")
        return list(dir_path.glob(pattern))


    def check_if_data_exists(self, dir_key, file_name):
        """
        Load data from a file. Cache it for faster subsequent loads.
        """
        file_path = self.directories[dir_key] / file_name
        return file_path.exists()

    def save_data(self, dir_key, file_name, data, use_cache=True): # PM.save_data(dir_key, file_name, data)
        """
        Save data to a file.
        """
        file_path = self.directories[dir_key] / file_name
        # print(f"saving data to file_path: {file_path}, with data of type: {type(data)}")
        if file_path.suffix == ".csv":
            if isinstance(data, dict):
                pd.DataFrame.from_dict(data).to_csv(file_path, index=False)
            elif isinstance(data,pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError(f"tried to save csv with an unsuppported data type: {type(data)}")
        elif file_path.suffix == ".npy":
            np.save(file_path, data)
        elif file_path.suffix == ".txt" and isinstance(data,list):
            with open(file_path, 'w') as f:
                for x in data:
                    f.write(x + '\n')
        elif file_path.suffix == ".txt":
            with open(file_path, 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        # Update cache
        if use_cache:
            self.file_cache[file_name] = data


    def load_data(self, dir_key, file_name, use_cache=False):
        """
        Load data from a file. Cache it for faster subsequent loads.
        """
        file_path = self.directories[dir_key] / file_name
        
        if use_cache and file_name in self.file_cache:
            return self.file_cache[file_name]
        
        if not file_path.exists():
            return None
        
        if file_path.suffix == ".csv":
            data = pd.read_csv(file_path)
        elif file_path.suffix == ".npy":
            data = np.load(file_path)
        elif file_path.suffix == ".txt" and "layers" in file_name:
            with open(file_path, 'r') as f:
                data = [line.strip() for line in f]
        elif file_path.suffix == ".txt":
            with open(file_path, 'r') as f:
                data = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
        if use_cache:
            self.file_cache[file_name] = data
        return data

    def save_dataset(self, dir_key, custom_name=None, data=None):
        """
        Save non-model-specific datasets using standardized naming conventions.
        """
        if dir_key not in ["word_features", "sentence_features", "gpt_input"]:
            raise ValueError(f"Unsupported directory key for saving dataset: {dir_key}")

        # Use custom name if provided, else use a standardized name
        file_name = custom_name if custom_name else f"{dir_key}_dataset.csv"
        self.save_data(dir_key, file_name, data)

    def load_dataset(self, dir_key, custom_name=None):
        """
        Load non-model-specific datasets using standardized naming conventions.
        """
        if dir_key not in ["word_features", "sentence_features", "gpt_input"]:
            raise ValueError(f"Unsupported directory key for loading dataset: {dir_key}")

        # Use custom name if provided, else use a standardized name
        if custom_name:
            return self.load_data(dir_key, custom_name)
        else:
            return self.load_csvs_as_dataframe(dir_key)

    def save_activations(self, activations, layers, model, metadata_df, use_cache=False): # PM.save_activations(activations, layers, model, metadata_df)
        # label, activations_fn, metadata_fn, layers_fn  = get_activation_names(model, version, activations_index)
        activations_fn, metadata_fn, layers_fn = self.generate_file_patterns("activations", model)
        PM.save_data("activations", activations_fn, activations,use_cache=use_cache)
        PM.save_data("activations", layers_fn, layers,use_cache=use_cache)
        PM.save_data("activations", metadata_fn, metadata_df,use_cache=use_cache)

    def load_activations(self, model, version=1, activations_indexes=[0], use_cache=False): # PM.load_activations(model)
        all_activations=[]
        for activations_index in activations_indexes:
            activations_fn, metadata_fn, layers_fn = self.generate_file_patterns("activations", model)
            # label, activations_fn, layers_fn, metadata_fn = get_activation_names(model, version, activations_index)
            all_activations.append(self.load_data("activations",activations_fn,use_cache=use_cache))
        activations=np.concatenate(all_activations,axis=1)
        # load layers
        layers=self.load_data("activations",layers_fn,use_cache=use_cache)
        metadata_df = self.load_data("activations", metadata_fn,use_cache=use_cache)
        return activations, layers, metadata_df


    # # sample_limit=10000
    # # if sample_limit and sample_limit < PM.sample_idx_max - PM.sample_idx_min: PM.sample_idx_max = PM.sample_idx_min + sample_limit
    # # PM.load_ys() # , use_cache=False)
    # # PM.load_Xss(all_models) # , use_cache=False)
    # ys_6 = PM.load_ys(compress_to_POS_6=True,v=0) # , use_cache=False)
    # Xss_6 = PM.load_Xss(all_models, compress_to_POS_6=True,v=0) # , use_cache=False)
    # if False: PM.standardize_Xss_ys()
    def load_Xss(self, models, use_cache=True, compress_to_POS_6=False,v=1):

        if not isinstance(models,list): models=[models]
        for model in tqdm(models, desc='loading models'):
            if compress_to_POS_6:
                if use_cache and model in list(self.Xss_6): continue  # skip if compressed model already loaded
            else:
                if use_cache and model in list(self.Xss): continue # skip if model already loaded
            Xs={}
            (activations, layers, metadata_df) = self.load_activations(model)
            # sample_idx_min = metadata_df["context_length"].idxmax()
            # sample_idx_max = len(metadata_df["context_length"])
            # print("sample_idxes",sample_idx_min,sample_idx_max)
            for i in range(len(layers)):
                Xs[layers[i]]=activations[i][self.sample_idx_min:self.sample_idx_max]
                if compress_to_POS_6:
                    Xs[layers[i]]=Xs[layers[i]].compress(self.is_in_POS_6,axis=0)
                # Xs[layers[i]]=standardize_np_array(activations[i][sample_idx_min:sample_idx_max])
            if compress_to_POS_6:
                self.Xss_6[model]=Xs
            else:
                self.Xss[model]=Xs

        Xss=self.Xss
        Xss_6=self.Xss_6

        if v:
            if compress_to_POS_6:
                print(f"all models: {list(Xss_6.keys())}")
                print("="*4)
                print(f"{dFirst(dFirst(Xss_6)).shape[0]: <4} = number of samples")
                print(f"{len(Xss_6): <4} = number of models")
                print(f"{len(dFirst(Xss_6)): <4} = number of layers in 1st model")
                print(f"Xss_6 shape: {dFirst(dFirst(Xss_6)).shape}")
                print(f"Size of Xss_6: {str(get_size_of_dict(Xss_6))} GB")
                print(f"Size of Xss: {str(get_size_of_dict(Xss))} GB")
                print("="*100)
            else:
                print(f"all models: {list(Xss.keys())}")
                print("="*4)
                print(f"{dFirst(dFirst(Xss)).shape[0]: <4} = number of samples")
                print(f"{len(Xss): <4} = number of models")
                print(f"{len(dFirst(Xss)): <4} = number of layers in 1st model")
                print(f"X shape: {dFirst(dFirst(Xss)).shape}")
                print(f"Size of Xss: {str(get_size_of_dict(Xss))} GB")
                print("="*100)

        if compress_to_POS_6:
            return self.Xss_6
        else:
            return self.Xss


    def load_ys(self, use_cache=True, compress_to_POS_6=False,v=1):
        if use_cache and self.ys:
            return
        # sample_idx_min = self.sample_idx_min # 386
        # sample_idx_max = self.sample_idx_max # 8344
        # sentence_level_dict = load_sentence_level_dict()
        # word_level_dict = load_word_level_dict()
        word_features = self.load_dataset("word_features")
        # self.word_features = word_features
        word_level_dict = word_features.to_dict(orient="list")
        mydatadict={x: np.array(word_level_dict[x][self.sample_idx_min:self.sample_idx_max]) for x in tqdm(word_level_dict.keys(), desc='loading mydatadict')}
        # mydataset=load_from_disk('mydata_with_index.hf')
        # deal with rolled data labels
        for x in self.classification_labels + self.reggression_labels + self.pos_labels:
            if x[-1] == "+": mydatadict[x] = np.roll(mydatadict[x[:-1]], -1)
            if x[-1] == "-": mydatadict[x] = np.roll(mydatadict[x[:-1]], 1)

        # self.ys = dict([(x, mydatadict[x]) for x in self.classification_labels + self.reggression_labels + self.pos_labels]) # ys_reggression, ys_classification
        self.is_in_POS_6 = mydatadict["is_in_POS_6"]
        # for x in self.classification_labels + self.reggression_labels + self.pos_labels:
        for x in mydatadict.keys():
            self.ys[x] = mydatadict[x]
            if compress_to_POS_6:
                self.ys_6[x]=self.ys[x].compress(self.is_in_POS_6,axis=0)

        ys=self.ys
        ys_6=self.ys_6

        if v:
            if compress_to_POS_6:
                print(f"{len(ys_6): <4} = number of possible targets")
                print(f"y_6 shape: {dFirst(ys_6).shape}")
                print(f"Size of ys_6: {str(get_size_of_dict(ys_6))} GB")
                print(f"Size of ys: {str(get_size_of_dict(ys))} GB")
                print("="*100)
                print(f"\nclassification_labels: ", end="")
                print_d({k:ys_6[k] for k in self.classification_labels})
                print(f"\nreggression_labels: ", end="")
                print_d({k:ys_6[k] for k in self.reggression_labels})
                print(f"\npos_labels: ", end="")
                print_d({k:ys_6[k] for k in self.pos_labels}); 
                print(f"\other_labels: ", end="")
                print_d({k:ys_6[k] for k in ys_6 if not k in self.pos_labels + self.reggression_labels + self.classification_labels})
            else:
                print(f"{len(ys): <4} = number of possible targets")
                print(f"y shape: {dFirst(ys).shape}")
                print(f"Size of ys: {str(get_size_of_dict(ys))} GB")
                print("="*100)
                print(f"\nclassification_labels: ", end="")
                print_d({k:ys[k] for k in self.classification_labels})
                print(f"\nreggression_labels: ", end="")
                print_d({k:ys[k] for k in self.reggression_labels})
                print(f"\npos_labels: ", end="")
                print_d({k:ys[k] for k in self.pos_labels}); 
                print(f"\other_labels: ", end="")
                print_d({k:ys[k] for k in ys if not k in self.pos_labels + self.reggression_labels + self.classification_labels})

        if compress_to_POS_6:
            return self.ys_6
        else:
            return self.ys

    def standardize_Xss_ys(self):
        for k,v in self.ys.items():
            if k in self.reggression_labels:
                self.ys[k]=standardize_np_array(v)
            else:
                self.ys[k]=v
        for k,v in self.ys_6.items():
            if k in self.reggression_labels:
                self.ys_6[k]=standardize_np_array(v)
            else:
                self.ys_6[k]=v
        for model,Xs in self.Xss.items():
            for layer, X in Xs.items():
                self.Xss[model][layer]=standardize_np_array(X)
        for model,Xs in self.Xss_6.items():
            for layer, X in Xs.items():
                self.Xss_6[model][layer]=standardize_np_array(X)

    # def save_analysis(self, model, version=1, activations_indexes=[0]): # PM.load_activations(model)
    # def load_analysis(self, model, version=1, activations_indexes=[0]): # PM.load_activations(model)

    def load_csvs_as_dataframe(self, dir_key, pattern="*.csv"):
        """
        Load all CSVs in a directory into a single dataframe.
        """
        files = self.get_files(dir_key, pattern)
        dfs = [pd.read_csv(file, low_memory=False) for file in files]
        # return pd.concat(dfs, ignore_index=True)
        return pd.concat(dfs, axis=1)

    def load_all(self, dir_key):
        """
        Load all files from a specified directory into the cache.
        """
        files = self.get_files(dir_key)
        res=[]
        for file in files:
            # Use the absolute path as the key for cache for uniqueness
            absolute_path = file.resolve()
            if file.suffix == ".csv":
                data = pd.read_csv(file)
            elif file.suffix == ".npy":
                data = np.load(file)
            elif file.suffix == ".txt":
                with open(file, 'r') as f:
                    data = f.read()
            else:
                print(f"Warning: Unsupported file type {file.suffix} for {file}. Skipping.")
                continue
            res.append(data)
            self.file_cache[str(absolute_path)] = data
        return res
