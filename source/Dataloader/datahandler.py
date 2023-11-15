import os
import csv
import PIL
import omegaconf
import numpy as np

from source.Dataloader import * # Fetch all datasets
from source.Logger.logger import *
from torch.utils.data import Dataset
from omegaconf.dictconfig import DictConfig


class DataHandler(Dataset):
    r"""
    # DataHandler

    Base class for custom datasets. Datasets are meant to load your data as you want and need. 
    Your custom dataset will be fetched by the agent automatically and used within the session
    batch-wise.
    """
    def __init__(self, cfg_dataset: DictConfig) -> None:
        r"""
        Args:
        
        * `cfg (omegaconf.DictConfig)`:
            * Hydra based configuration dictionary based on the given configuration .yaml file.
            **Only the subset cfg object for the dataset is available at this point.**
        """
        super().__init__()
        self.cfg = cfg_dataset

        ## Member of each dataset which will hold the information of all files, i.e. paths 
        ## to input images or/and labels.
        self.filenames = []

        ## If this dataset is not only used for training but also used for validation (so they is
        ## no separate dataset defined), both member will hold the information which indices of
        ## `self.filenames` belong to the training and validation set.
        self.indices_train = []
        self.indices_valid = []

        self.root = self.cfg.root

        try:
            self.aug_methods = self.cfg.augmentation_methods
        except omegaconf.errors.ConfigAttributeError:
            self.aug_methods = None

        if self.aug_methods == None or len(self.aug_methods) == 0:
            log_info("[Dataset/{}] No augmentation is done.".format(self.cfg.name))
        else:
            log_info("[Dataset/{}] Following augmentation are being used: {}.".format(self.cfg.name, ", ".join(self.aug_methods)))

        self.search_files()
        self.validate_files()
        log_info("[Dataset/{}] Found {} entries.".format(self.cfg.name, len(self.filenames)))

    def __len__(self) -> int:
        r""" Returns the size of the dataset according to the number of found files. """
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict:
        r""" 
        **Abstract**: Returns current item according to given index. This needs to be implemented by custom dataset. If not, an error is thrown.
        """
        raise NotImplementedError

    def search_files(self) -> None:
        r"""
        **Abstract**: Search for all files/data in a given root directory. Either search for all 
        information using a certain folder structure or use a filelist, defined in the 
        configuration file. This needs to be implemented by the custom dataset.
        """
        raise NotImplementedError

    def validate_files(self) -> None:
        r"""
        Validates that none of all found files is corrupted and can be read correctly. If not, 
        they will be removed from filenames beforehand.
        """
        remove_idxs = []
        for idx, file_tuple in enumerate(self.filenames):
            for file in file_tuple:
                ## Only images will be verified so far.
                ## PIL is used for verification. If no error is thrown
                ## when verify() is called it is considered to be not 
                ## corrupted. If an error is thrown, the file index will 
                ## be saved for removing of the list later.
                if not file.endswith(".png") or not file.endswith(".jpg"):
                    continue
                try:
                    im = PIL.Image.open(file)
                    im.verify()
                    im.close()
                except (IOError, OSError, PIL.Image.DecompressionBombError):
                    remove_idxs.append(idx)
                    break
        
        if len(remove_idxs) > 0:
            log_warning("[DataHandler/{}] Found {} corrupted file pairs. They will be removed from the dataset.".format(self.cfg.name, len(remove_idxs)))
            for idx in remove_idxs:
                del self.filenames[idx]

    def split_train_valid(self):
        r"""
        Splits the dataset into two different subsets that can be used for training and validation.
        This is done by shuffling the file set randomly and compute the separator according to the
        validation/train split defined with `self.cfg.validation_ratio`.
        """
        assert len(self) > 0, "[Dataset/{}] Split into subsets not possible \
                                            since dataset is empty.".format(self.cfg.name)

        try:
            _indices = list(range(len(self)))
            np.random.seed(42)
            np.random.shuffle(_indices)

            ## Split index list for training and validation split
            split_at         = int(np.floor((1. - self.cfg.validation_ratio) * len(self)))
            self.indices_train, self.indices_valid = _indices[:split_at], _indices[split_at+1:]

        except omegaconf.errors.ConfigAttributeError:
            log_error("[Dataset/{}] cfg.dataset.validation_ratio is not defined for splitting dataset into two subsets.".format(self.cfg.name))
        
        assert len(self.indices_train) > 0 or len(self.indices_valid) > 0, "[Dataset/{}] After subset split one array of indices must not be empty.".format(self.cfg.name)

    def dump_datahandler(self, out_dir: str, file_name: str):
        r"""
        Dumping out the information of the DataHandler like used data, splitted train and validation set.

        Args:
        
        * `out_dir (str)`: Path where the file is dumped to.
        * `filename (str)`: Name of the csv file.
        """
        ## Check if only this datahandler is used for train and validation by checking the length
        ## of the indices list. If they were not set (empty) another dataset was used originally 
        ## for validation.
        is_splitted = (len(self.indices_train) > 0) and (len(self.indices_valid) > 0)

        ## Dataset information can be dumped out only if `self.filenames` contains file paths so far.
        are_files = True
        for files_entry in self.filenames:
            for file in files_entry:
                if not os.path.isfile(file):
                    are_files = False
                    break
        if not are_files:
            log_info("[Dataset/{}] Cannot dump out information since self.files does not contain file information.".format(self.cfg.name))
        
        else:
            with open(os.path.join(out_dir, file_name), 'w', newline='') as f:
                writer = csv.writer(f)
                if is_splitted:
                    writer.writerow(["Training"])
                    writer.writerows([self.filenames[i] for i in self.indices_train])
                    writer.writerow([])
                    writer.writerow(["Validation"])
                    writer.writerows([self.filenames[i] for i in self.indices_valid])
                else:
                    writer.writerows(self.filenames)

    def finalize(self, out_dir: str, file_name: str):
        r""" Finalizing step for all datasets that includes dumping out information about the dataset itself. """
        self.dump_datahandler(out_dir=out_dir, file_name=file_name)


class MultiDataHandler(Dataset):
    r"""
    # MultiDataHandler

    Base class for custom multi datasets that can merge multiple datasets into one. Datasets are
    meant to load your data as you want and need. Your custom dataset will be fetched by the agent
    automatically and used within the session batch-wise.
    """
    def __init__(self, cfg: DictConfig, datasets: dict) -> None:
        r"""
        Args:
        
        * `cfg (omegaconf.DictConfig)`: 
            * Hydra based configuration dictionary based on the given configuration .yaml file.
            **Only the subset cfg object for the dataset is available at this point.**
        * `datasets (dict)`:
            * Dictionary that stores all datasets to create a MultiDatahandler
              out of it. For each key (dataset name) a class object is given 
              as value.
        """
        super().__init__()
        self.cfg = cfg
        self.datasets = datasets

        ## Create mapping from given index to a specfic dataset.
        self.idx2dataset = [(d_str, d_idx) for d_str in self.datasets.keys() for d_idx in range(len(self.datasets[d_str]))]
        assert len(self) == len(self.idx2dataset), \
            "[MultiDataHandler] Number of Index to Dataset mapping does not match length of MultiDataset."

        ## If this dataset is not only used for training but also used for validation (so they is
        ## no separate dataset defined), both member will hold the information which indices of
        ## `self.filenames` belong to the training and validation set.
        self.indices_train = []
        self.indices_valid = []

    def __len__(self) -> int:
        r""" Returns size of dataset according to the number of found files. """
        _len = 0
        for _, v in self.datasets.items():
            _len = _len + len(v)
        return _len

    def __getitem__(self, index: int) -> dict:
        r""" 
        **Abstract**: Fetches current item according to given index. This needs to be implemented
        by custom dataset. If not, an error is thrown.
        """
        raise NotImplementedError


    def split_train_valid(self):
        r"""
        Splits the dataset into two different subsets that can be used for training and validation.
        This is done by shuffling the file set randomly and compute the separator according to the
        validation/train split defined with `self.cfg.validation_ratio`.
        """
        assert len(self) > 0, "[DataHandel] Split into subsets not possible \
                                            since dataset is empty."

        try:
            _indices = list(range(len(self)))
            np.random.seed(42)
            np.random.shuffle(_indices)

            ## Split index list for training and validation split
            split_at         = int(np.floor((1. - self.cfg.validation_ratio) * len(self)))
            self.indices_train, self.indices_valid = _indices[:split_at], _indices[split_at+1:]

        except omegaconf.errors.ConfigAttributeError:
            log_error("[DataHandler] cfg.dataset.validation_ratio is not defined for splitting dataset into two subsets.")
        
        assert len(self.indices_train) > 0 or len(self.indices_valid) > 0, \
            "[DataHandler] After subset split one array of indices must not be empty."

    def dump_datahandler(self, out_dir: str, file_name: str):
        r"""
        Dumping out the information of the DataHandler like used data, splitted train and validation set.

        Args:
        
        * `out_dir (str)`: Path where the file is dumped to.
        * `filename (str)`: Name of the csv file.
        """
        ## Check if only this datahandler is used for train and validation by checking the length
        ## of the indices list. If they were not set (empty) another dataset was used originally
        ## for validation.
        is_splitted = (len(self.indices_train) > 0) and (len(self.indices_valid) > 0)

        ## Dataset information can be dumped out only if `self.filenames` of all datasets contain file paths.
        for _, v in self.datasets.items():
            for files_entry in v.filenames:
                for file in files_entry:
                    if not os.path.isfile(file):
                        log_info("[DataHandler] Cannot dump out information since {} does not contain file information.".format(file))
                        return

        with open(os.path.join(out_dir, file_name), 'w', newline='') as f:
            writer = csv.writer(f)
            if is_splitted:
                writer.writerow(["Training"])
                writer.writerows([self.datasets[self.idx2dataset[i][0]].filenames[self.idx2dataset[i][1]] for i in self.indices_train])
                writer.writerow([])
                writer.writerow(["Validation"])
                writer.writerows([self.datasets[self.idx2dataset[i][0]].filenames[self.idx2dataset[i][1]] for i in self.indices_valid])
            else:
                writer.writerows(self.filenames)

    def finalize(self, out_dir: str, file_name: str):
        r""" Finalizing step for all datasets that includes dumping out information about the dataset itself. """
        self.dump_datahandler(out_dir=out_dir, file_name=file_name)
