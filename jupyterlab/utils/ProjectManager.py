import os
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
from utils.my_utils import *


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
    Manages project directories, data loading, saving operations, and mappings for models and Part-of-Speech (POS) tags.

    Attributes:
        Xss (dict): A dictionary storing activation data for various models and layers.
        Xss_6 (dict): Similar to `Xss`, but with data compressed to the POS-6 level of granularity.
        ys (dict): A dictionary storing target variables for various classification and regression tasks.
        ys_6 (dict): Similar to `ys`, but with data compressed to the POS-6 level.
        is_in_POS_6 (list): A boolean list indicating which data points are included in the POS-6 tagset.

        file_cache (dict): A cache of loaded files to prevent reloading data multiple times, improving performance.

        base_dir (Path): The base directory for the project data, defaulting to the user's home directory.
        directories (dict): A mapping of data categories (e.g., 'transformer_weights', 'word_features') to their corresponding directories within the project.

        fn_base_sent_features (str): Filename for base sentence features.
        fn_base_word_features (str): Filename for base word features.
        fn_word_indexes (str): Filename for word indexes.
        fn_pos_features (str): Filename for POS features.
        fn_word_frequencies (str): Filename for word frequencies.
        fn_tree_depth (str): Filename for tree depth data.
        fn_function (str): Filename for function data (e.g., function vs. content words).

        sample_idx_min (int): Starting word index of conll2012_ontonotesv5 sample.
        sample_idx_max (int): Ending word index of conll2012_ontonotesv5 sample.
        sels (list): A list of indices used in analysis.
        maxps (list): A list of indices used in analysis.

        all_models (list): A list of all models used in the project (e.g., 'gpt2', 'gpt2-untrained').
        gpt2_models (list): List of GPT-2 models used in the project.
        gpt2xl_models (list): List of GPT-2 XL models used in the project.
        model_label_map (dict): A mapping between internal model names and their display labels (e.g., 'gpt2-xl' -> 'XL-Trained').
        model_group_map (dict): A mapping of models to their respective training groups (e.g., 'Trained', 'Untrained', 'Gaussian') for analysis.
        model_groups (dict): A mapping of models to their respective wieght groups (e.g., 'main', 'single', 'doubles', and 'quads') for analysis.

        layers_gpt2xl (list): A list of layer names for the GPT-2 XL model, including the "drop" layer.
        layers_gpt2 (list): A list of layer names for the GPT-2 model, derived from `layers_gpt2xl`.
        layers_dict (dict): A mapping of models to their respective layer lists (e.g., GPT-2 or GPT-2 XL layers).
        layers_to_idx (dict): A dictionary mapping layer names to their indices for easier lookup.

        POS_51_all_tags (list): A list of all 51 POS tags used in the project.
        POS_12_all_tags (list): A list of 12 broader POS tags for a higher-level classification.
        POS_7_all_tags (list): A list of 7 even broader POS tags, with some tags collapsed into categories like 'X' for unknown or unclassifiable tags.
        POS_6_all_tags (list): A list of 6 POS tags, representing the most basic categorization (e.g., Noun, Verb).

        POS_12_to_POS_7 (dict): A mapping from the 12-tag POS set to the 7-tag set (e.g., 'NOUN' -> 'Noun').
        POS_51_tag_to_id (dict): A mapping from the 51-tag POS set to their corresponding indices.
        POS_12_tag_to_id (dict): A mapping from the 12-tag POS set to their corresponding indices.
        POS_7_tag_to_id (dict): A mapping from the 7-tag POS set to their corresponding indices.
        POS_6_tag_to_id (dict): A mapping from the 6-tag POS set to their corresponding indices.

        pos_names (list): POS tag names

        classification_labels (list): A list of all classification labels.
        my_classification_targets (list): The subset of the classification labels used for analysis. (and for Next, Current, and Previous words)
        reggression_labels (list): Labels for regression tasks. (and for Next, Current, and Previous words)
        pos_labels (list): All part-of-speech targets for classification tasks. (and for Next, Current, and Previous words)




    """

    def __init__(self):
        """
        Initializes the ProjectManager with default settings, directories, 
        model mappings, and POS tag configurations.
        """
        # Initialize model activations and targets dictionaries
        self.Xss = {}
        self.ys = {}
        self.ys_6 = {}
        self.Xss_6 = {}
        self.is_in_POS_6 = []

        # File caching for quicker data access
        self.file_cache = {}

        # Base directory for project data (default to user's HOME directory)
        self.base_dir = Path(os.getenv("HOME")) / "data"

        # Initialize directory structure for various data categories
        self.directories = {
            "transformer_weights": self.base_dir / "transformer_weights",
            "word_features": self.base_dir / "word_features",
            "sentence_features": self.base_dir / "sentence_features",
            "scores": self.base_dir / "scores",
            "activations": self.base_dir / "activations",
            "context_lengths": self.base_dir / "context_lengths",
            "analysis_res": self.base_dir / "analysis_res",
            "thesis_tex": self.base_dir / "thesis_tex",
            "tbl": self.base_dir / "thesis_tex" / "tbl",
            "fig": self.base_dir / "thesis_tex" / "fig",
            "fimg": self.base_dir / "thesis_tex" / "fimg",
            "hf_files": self.base_dir / "hf_files",
            "gpt_input": self.base_dir / "gpt_input",
            "exp1": self.base_dir / "experiment_1_results",
            "exp2": self.base_dir / "experiment_2_results",
        }

        # Initialize filenames for specific datasets
        self.fn_base_sent_features = "base_sentence_features.csv"
        self.fn_base_word_features = "base_word_features.csv"
        self.fn_word_indexes = "word_indexes.csv"
        self.fn_pos_features = "pos_features.csv"
        self.fn_word_frequencies = "word_frequencies.csv"
        self.fn_tree_depth = "tree_depth.csv"
        self.fn_function = "function.csv"

        # Sample index boundaries
        self.sample_idx_min = 386  # Starting index for samples
        self.sample_idx_max = 8344  # Ending index for samples

        # Selection indices for analysis
        self.sels = [f"sel_{i}" for i in range(6)]
        self.maxps = [f"maxp_{i}" for i in range(6)]

        # Initialize models, layers, and label mappings
        self._initialize_model_mappings()

        # Initialize POS tags and mappings
        self._initialize_pos_tags()

        # Initialize labels for regression and classification targets
        self._initialize_analysis_labels()

    def _initialize_model_mappings(self):
        """
        Initializes model labels, layer mappings, and groupings for different models.
        """
        # Define all models and map their internal labels to display labels
        self.all_models = [
            'gpt2-xl', 'gpt2-xl-untrained_1', 'gpt2',
            *[f"gpt2-untrained_{i}" for i in range(1, 10)],
            *[f"gpt2-untrained_{i}_weight_config_all" for i in range(1, 10)]
        ]
        self.gpt2xl_models = self.all_models[:2]  # First two models are GPT-2 XL
        self.gpt2_models = self.all_models[2:]    # Remaining models are GPT-2
        # Mapping between model labels and their display names
        self.model_label_map = {
            'gpt2-xl': 'XL-Trained',
            'gpt2-xl-untrained_1': 'XL-Untrained_1',
            'gpt2': 'Trained',
            **{f"gpt2-untrained_{i}": f"Untrained_{i}" for i in range(1, 10)},
            **{f"gpt2-untrained_{i}_weight_config_all": f"Gaussian_{i}" for i in range(1, 10)}
        }

        # Grouping models for structured analysis
        self.model_group_map = {
            'gpt2-xl': 'XL-Trained',
            'gpt2-xl-untrained_1': 'XL-Untrained',
            'gpt2': 'Trained',
            **{f"gpt2-untrained_{i}": 'Untrained' for i in range(1, 10)},
            **{f"gpt2-untrained_{i}_weight_config_all": 'Gaussian' for i in range(1, 10)},
        }

        # Model wight grouping setup (for now deleted expiriment)
        self.model_groups = {
            "main": ["gpt2"] + [f"gpt2-untrained_{i}" for i in range(10)] + ["all"],
            "single": [f"single_{i}" for i in range(12)],
            "doubles": [f"double_{i}" for i in range(6)],
            "quads": ['attns', 'mlps', 'lns'],
        }

        # Layer definitions for each model
        self.layers_gpt2xl = ["drop"] + [f'encoder.h.{i}' for i in range(48)]
        self.layers_gpt2 = self.layers_gpt2xl[:13]

        # Map models to their respective layer configurations
        self.layers_dict={
            **{model:self.layers_gpt2xl for model in self.gpt2xl_models},
            **{model:self.layers_gpt2 for model in self.gpt2_models}
            }

        # Mapping from layer names to their indices
        self.layers_to_idx = {layer: idx for idx, layer in enumerate(self.layers_gpt2xl)}

    def _initialize_pos_tags(self):
        """
        Initializes Part-of-Speech (POS) tags and related mappings for various levels of granularity.
        """
        # POS tag levels and their mappings for classification tasks
        self.POS_51_all_tags = [
            "XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", "HYPH", "IN",
            "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", "PRP$", "RB", "RBR", "RBS", "RP",
            "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", "WDT", "WP", "WP$", "WRB"
        ]
        self.POS_12_all_tags = ["VERB", "NOUN", "PRON", "ADJ", "ADV", "ADP", "CONJ", "DET", "NUM", "PRT", "X", "."]
        self.POS_7_all_tags = ["Noun", "Verb", "Adposition", "Determiner", "Adjective", "Adverb", "X"]
        self.POS_6_all_tags = ["Noun", "Verb", "Adposition", "Determiner", "Adjective", "Adverb"]

        # Mappings from broader to narrower POS tags
        self.POS_12_to_POS_7 = {
            "NOUN": "Noun", "PRON": "Noun", "VERB": "Verb", "ADP": "Adposition",
            "DET": "Determiner", "ADJ": "Adjective", "ADV": "Adverb", "CONJ": "X",
            "NUM": "X", "PRT": "X", "X": "X", ".": "X"
        }

        # Tag to ID mappings for easy lookup
        self.POS_51_tag_to_id = {tag: idx for idx, tag in enumerate(self.POS_51_all_tags)}
        self.POS_12_tag_to_id = {tag: idx for idx, tag in enumerate(self.POS_12_all_tags)}
        self.POS_7_tag_to_id = {tag: idx for idx, tag in enumerate(self.POS_7_all_tags)}
        self.POS_6_tag_to_id = {tag: idx for idx, tag in enumerate(self.POS_6_all_tags)}

        # POS tag names
        self.pos_names = ["XX", "``", "$", "''", "*", ",", "-LRB-", "-RRB-", ".", ":", "ADD", "AFX", "CC", "CD", "DT", "EX", "FW", 
                          "HYPH", "IN", "JJ", "JJR", "JJS", "LS", "MD", "NFP", "NN", "NNP", "NNPS", "NNS", "PDT", "POS", "PRP", 
                          "PRP$", "RB", "RBR", "RBS", "RP", "SYM", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", "VERB", 
                          "WDT", "WP", "WP$", "WRB"]

    def _initialize_analysis_labels(self):
        """
        Initialize label names for regression and classification targets
        """

        # Classification labels
        self.classification_labels = []
        for x in ['words', 'word_idx', 'predicate_lemmas', 'predicate_framenet_ids', 'word_senses', 'named_entities', 'function', 'tree_depth']:
            self.classification_labels += [x, f"{x}-", f"{x}+"]

        # Custom classification target labels
        self.my_classification_targets = []
        for x in ['function', 'tree_depth', 'word_idx']:
            self.my_classification_targets += [x, f"{x}-", f"{x}+"]

        # Regression labels
        self.reggression_labels = []
        for x in ['sentence_idx', 'unigram_probs', 'bigram_probs', 'trigram_probs']:
            self.reggression_labels += [x, f"{x}-", f"{x}+"]

        # POS labels for various tasks
        self.pos_labels = []
        for x in ['pos_tags', 'POS_12_id', 'POS_7_id']:
            self.pos_labels += [x, f"{x}-", f"{x}+"]




    def _get_model_label(self, display_label):
        """
        Convert a display label to the corresponding internal model label.

        Args:
            display_label (str): The display label of the model.

        Returns:
            str: The internal model label corresponding to the display label.

        Raises:
            ValueError: If the display label is not recognized.
        """
        for model_label, disp_label in self.model_label_map.items():
            if disp_label == display_label or model_label == display_label:
                return model_label
        raise ValueError(f"Unknown display label: {display_label}")


    def _get_display_label(self, model_label):
        """
        Convert an internal model label to the corresponding display label.

        Args:
            model_label (str): The internal model label.

        Returns:
            str: The display label corresponding to the model label.

        Raises:
            ValueError: If the model label is not recognized.
        """
        for _model_label, display_label in self.model_label_map.items():
            if model_label == _model_label or model_label == display_label:
                return display_label
        raise ValueError(f"Unknown display label: {display_label}")


    def get_base_and_full_labels(self, model_label):
        """
        Get the base and full labels for a given model.

        Args:
            model_label (str): The internal model label.

        Returns:
            tuple: A tuple of (base_label, full_label) for the model.
        """
        if model_label in ["gpt2", "gpt2-untrained", "gpt2-xl", "gpt2-xl-untrained"]:
            base_label = model_label
            full_label = model_label
        elif "_weight_config_" in model_label:
            base_label = "gpt2-untrained"
            full_label = f"{model_label}"
        else:
            base_label = "gpt2-untrained"
            full_label = f"{base_label}_weight_config_{model_label}"
        return base_label, full_label


    def check_for_existing_patterns(self, dir_key):
        """
        Check for existing file patterns in the specified directory.

        Args:
            dir_key (str): The directory key.

        Returns:
            list: A list of models for which file patterns exist in the directory.
        """
        return [x for x in self.all_models if self.do_patterns_exist(dir_key, x)]


    def do_patterns_exist(self, dir_key, model_label):
        """
        Check if file patterns for a given model exist in the specified directory.

        Args:
            dir_key (str): The directory key.
            model_label (str): The internal model label.

        Returns:
            bool: True if file patterns exist, False otherwise.
        """
        files = self.generate_file_patterns(dir_key, model_label)
        if isinstance(files, tuple):
            files = list(files)
        else:
            files = [files]
        return all(Path(self.directories[dir_key], f).is_file() for f in files)


    def generate_file_patterns(self, dir_key, model_label):
        """
        Generate file patterns for a given model label and directory key.

        Args:
            dir_key (str): The directory key (e.g., 'activations').
            model_label (str): The internal model label.

        Returns:
            str or tuple: The file pattern(s) associated with the given model and directory.
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
        else:
            raise ValueError(f"Unsupported directory key for pattern generation: {dir_key}")


    def fetch_data_by_label(self, dir_key, display_label):
        """
        Fetch data based on the directory key and display label.

        Args:
            dir_key (str): The directory key (e.g., 'activations').
            display_label (str): The display label of the model.

        Returns:
            list: A list of data files corresponding to the specified model and directory.
        """
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

        Args:
            dir_key (str): The directory key.
            pattern (str): The pattern to match files (default is '*').

        Returns:
            list: A list of files matching the pattern in the specified directory.
        """
        dir_path = self.directories.get(dir_key)
        if not dir_path:
            raise ValueError(f"Unknown directory key: {dir_key}")
        return list(dir_path.glob(pattern))


    def check_if_data_exists(self, dir_key, file_name):
        """
        Check if a data file exists in the specified directory.

        Args:
            dir_key (str): The directory key.
            file_name (str): The name of the file.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        file_path = self.directories[dir_key] / file_name
        return file_path.exists()


    def save_data(self, dir_key, file_name, data, use_cache=True):
        """
        Save data to a file in the specified directory.

        Args:
            dir_key (str): The directory key.
            file_name (str): The name of the file to save.
            data: The data to save.
            use_cache (bool): Whether to cache the file after saving (default is True).

        Raises:
            ValueError: If the file type is unsupported.
        """
        file_path = self.directories[dir_key] / file_name
        if file_path.suffix == ".csv":
            if isinstance(data, dict):
                pd.DataFrame.from_dict(data).to_csv(file_path, index=False)
            elif isinstance(data, pd.DataFrame):
                data.to_csv(file_path, index=False)
            else:
                raise ValueError(f"Unsupported data type for CSV: {type(data)}")
        elif file_path.suffix == ".npy":
            np.save(file_path, data)
        elif file_path.suffix == ".txt" and isinstance(data, list):
            with open(file_path, 'w') as f:
                for x in data:
                    f.write(x + '\n')
        elif file_path.suffix == ".txt":
            with open(file_path, 'w') as f:
                f.write(data)
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        if use_cache:
            self.file_cache[file_name] = data


    def load_data(self, dir_key, file_name, use_cache=False):
        """
        Load data from a file, with optional caching.

        Args:
            dir_key (str): The directory key.
            file_name (str): The name of the file to load.
            use_cache (bool): Whether to use cached data if available (default is False).

        Returns:
            The loaded data, or None if the file does not exist.

        Raises:
            ValueError: If the file type is unsupported.
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
        Save non-model-specific datasets to a file using standardized naming conventions.

        Args:
            dir_key (str): The directory key (e.g., 'word_features', 'sentence_features', 'gpt_input').
            custom_name (str, optional): A custom filename for the dataset (default is None).
            data: The dataset to save.

        Raises:
            ValueError: If the directory key is unsupported.
        """
        if dir_key not in ["word_features", "sentence_features", "gpt_input"]:
            raise ValueError(f"Unsupported directory key for saving dataset: {dir_key}")

        file_name = custom_name if custom_name else f"{dir_key}_dataset.csv"
        self.save_data(dir_key, file_name, data)


    def load_dataset(self, dir_key, custom_name=None):
        """
        Load non-model-specific datasets from a file using standardized naming conventions.

        Args:
            dir_key (str): The directory key (e.g., 'word_features', 'sentence_features', 'gpt_input').
            custom_name (str, optional): A custom filename for the dataset (default is None).

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            ValueError: If the directory key is unsupported.
        """
        if dir_key not in ["word_features", "sentence_features", "gpt_input"]:
            raise ValueError(f"Unsupported directory key for loading dataset: {dir_key}")

        if custom_name:
            return self.load_data(dir_key, custom_name)
        else:
            return self.load_csvs_as_dataframe(dir_key)


    def save_activations(self, activations, layers, model, metadata_df, use_cache=False):
        """
        Save model activations, layers, and metadata to files.

        Args:
            activations: The activations data to save.
            layers (list): The layers associated with the activations.
            model (str): The model label.
            metadata_df (pd.DataFrame): The metadata for the activations.
            use_cache (bool): Whether to cache the saved files (default is False).
        """
        activations_fn, metadata_fn, layers_fn = self.generate_file_patterns("activations", model)
        self.save_data("activations", activations_fn, activations, use_cache=use_cache)
        self.save_data("activations", layers_fn, layers, use_cache=use_cache)
        self.save_data("activations", metadata_fn, metadata_df, use_cache=use_cache)


    def load_activations(self, model, version=1, activations_indexes=[0], use_cache=False):
        """
        Load model activations, layers, and metadata from files.

        Args:
            model (str): The model label.
            version (int, optional): The version number of the activations (default is 1).
            activations_indexes (list, optional): The indexes of the activations to load (default is [0]).
            use_cache (bool): Whether to use cached activations if available (default is False).

        Returns:
            tuple: A tuple containing the activations (np.array), layers (list), and metadata (pd.DataFrame).
        """
        all_activations = []
        for activations_index in activations_indexes:
            activations_fn, metadata_fn, layers_fn = self.generate_file_patterns("activations", model)
            all_activations.append(self.load_data("activations", activations_fn, use_cache=use_cache))

        activations = np.concatenate(all_activations, axis=1)
        layers = self.load_data("activations", layers_fn, use_cache=use_cache)
        metadata_df = self.load_data("activations", metadata_fn, use_cache=use_cache)
        return activations, layers, metadata_df


    def load_Xss(self, models, use_cache=True, compress_to_POS_6=False, v=1):
        """
        Load activations for multiple models into the Xss attribute.

        Args:
            models (list or str): A list of model labels or a single model label to load.
            use_cache (bool, optional): Whether to use cached data if available (default is True).
            compress_to_POS_6 (bool, optional): Whether to compress the data to the POS-6 level (default is False).
            v (int, optional): Verbosity level for printing information (default is 1).

        Returns:
            dict: A dictionary of loaded activations.
        """
        if not isinstance(models, list):
            models = [models]

        for model in tqdm(models, desc='loading models'):
            if compress_to_POS_6:
                if use_cache and model in list(self.Xss_6):
                    continue  # skip if compressed model already loaded
            else:
                if use_cache and model in list(self.Xss): continue # skip if model already loaded
            Xs={}
            (activations, layers, metadata_df) = self.load_activations(model)
            # sample_idx_min = metadata_df["context_length"].idxmax()
            # sample_idx_max = len(metadata_df["context_length"])
            # print("sample_idxes",sample_idx_min,sample_idx_max)
            for i in range(len(layers)):
                Xs[layers[i]] = activations[i][self.sample_idx_min:self.sample_idx_max]
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


    def load_ys(self, use_cache=True, compress_to_POS_6=False, v=1):
        """
        Load target variables into the ys attribute, with optional compression to POS-6.

        Args:
            use_cache (bool, optional): Whether to use cached data if available (default is True).
            compress_to_POS_6 (bool, optional): Whether to compress the data to the POS-6 level (default is False).
            v (int, optional): Verbosity level for printing information (default is 1).

        Returns:
            dict: A dictionary of loaded target variables.
        """
        if use_cache and self.ys:
            return
        # sample_idx_min = self.sample_idx_min # 386
        # sample_idx_max = self.sample_idx_max # 8344
        # sentence_level_dict = load_sentence_level_dict()
        # word_level_dict = load_word_level_dict()
        word_features = self.load_dataset("word_features")
        # self.word_features = word_features
        word_level_dict = word_features.to_dict(orient="list")
        mydatadict = {x: np.array(word_level_dict[x][self.sample_idx_min:self.sample_idx_max]) for x in tqdm(word_level_dict.keys(), desc='loading mydatadict')}

        for x in self.classification_labels + self.reggression_labels + self.pos_labels:
            if x[-1] == "+":
                mydatadict[x] = np.roll(mydatadict[x[:-1]], -1)
            if x[-1] == "-":
                mydatadict[x] = np.roll(mydatadict[x[:-1]], 1)

        self.is_in_POS_6 = mydatadict["is_in_POS_6"]

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
        """
        Standardize the activations and target variables for models and layers.

        This function standardizes the values in `Xss` and `ys` attributes in place, using z-score normalization.
        """
        for k, v in self.ys.items():
            if k in self.reggression_labels:
                self.ys[k] = standardize_np_array(v)
            else:
                self.ys[k] = v

        for k, v in self.ys_6.items():
            if k in self.reggression_labels:
                self.ys_6[k] = standardize_np_array(v)
            else:
                self.ys_6[k] = v

        for model, Xs in self.Xss.items():
            for layer, X in Xs.items():
                self.Xss[model][layer] = standardize_np_array(X)

        for model, Xs in self.Xss_6.items():
            for layer, X in Xs.items():
                self.Xss_6[model][layer] = standardize_np_array(X)


    def load_csvs_as_dataframe(self, dir_key, pattern="*.csv"):
        """
        Load all CSV files from a directory into a single pandas DataFrame.

        Args:
            dir_key (str): The directory key.
            pattern (str, optional): The file pattern to match (default is "*.csv").

        Returns:
            pd.DataFrame: A concatenated DataFrame containing the data from all matching CSV files.
        """
        files = self.get_files(dir_key, pattern)
        dfs = [pd.read_csv(file, low_memory=False) for file in files]
        return pd.concat(dfs, axis=1)


    def load_all(self, dir_key):
        """
        Load all files from a directory into the cache and return them as a list.

        Args:
            dir_key (str): The directory key.

        Returns:
            list: A list of all loaded files.
        """
        files = self.get_files(dir_key)
        res = []
        for file in files:
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

