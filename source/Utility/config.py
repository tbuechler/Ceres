import os
import torch
import omegaconf

from source.Agent import *
from source.Utility.system import make_dirs
from source.Logger.logger import *
from omegaconf import open_dict


def process_config_file(cfg: omegaconf.DictConfig, root: str):
    r"""
    The experiment starts here, where also all experiment directories will be created and added to the DictConfig.

    Args:

    * `cfg (omegaconf.DictConfig)`: 
        * Class index tensor of shape (N, H, W). 
    * `root (str)`: 
        * Root of this project. 
    """

    with open_dict(cfg):
        cfg.experiment.summary_dir              = os.path.join(root, "experiments", cfg.experiment.name, "summaries/")
        cfg.experiment.checkpoint_dir           = os.path.join(root, "experiments", cfg.experiment.name, "checkpoints/")
        cfg.experiment.out_dir                  = os.path.join(root, "experiments", cfg.experiment.name, "out/")
        cfg.experiment.log_dir                  = os.path.join(root, "experiments", cfg.experiment.name, "logs/")

    make_dirs([cfg.experiment.summary_dir,          cfg.experiment.checkpoint_dir,
                cfg.experiment.out_dir,              cfg.experiment.log_dir])


def validate_configuration_file(cfg: omegaconf.DictConfig, mode: str):
    """
    Validates the correctness of some entries in the configuration file according to the current mode (train or eval).

    Args:

    * `cfg (omegaconf.DictConfig)`: 
        * Class index tensor of shape (N, H, W). 
    * `root (str)`: 
        * Root of this project. 
    """
    if cfg.experiment.name.__contains__(" "):
        cfg.experiment.name.replace(" ", "")
        log_info("[Config] Removed empty spaces in name: {}.".format(cfg.experiment.name))

    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(cfg.experiment.name))
        print(" *************************************** ")
    except omegaconf.errors.ConfigAttributeError:
        log_error("[Config] You must give this experiment a name by specifying cfg.experiment.name.")

    try:
        if cfg.experiment.on_device == 'cuda':
            if not torch.cuda.is_available():
                log_error("[Config] experiment.on_device was set to 'cuda' but the system does not support this!")
            else:
                log_info("[Config] Experiment runs on GPU (Cuda).")
        elif cfg.experiment.on_device == 'mps':
            if not torch.backends.mps.is_available():
                log_error("[Config] experiment.on_device was set to 'mps' but the system does not support this!")
            else:
                log_info("[Config] Experiment runs on MPS (Mac).")
        elif cfg.experiment.on_device == 'cpu':
            log_info("[Config] Experiment runs on CPU.")
        else:
            log_error("[Config] experiment.on_device (Currently: {}) must be specified with 'cpu', 'cuda' or 'mps'.".format(cfg.experiment.on_device))
    except omegaconf.errors.ConfigAttributeError:
        log_error("[Config] experiment.on_device must be specified with 'cpu', 'cuda' or 'mps'.")
        

    if mode == 'train':
        try:
            if cfg.experiment.overwrite_checkpoint:
                log_info("[Config] Checkpoint will always be overwritten after each epoch.")
            else:       
                log_info("[Config] A new checkpoint will always be created after each epoch.")
        except omegaconf.errors.ConfigAttributeError:
            with open_dict(cfg):
                cfg.experiment.overwrite_checkpoint = True
            log_info("[Config] Checkpoint will always be overwritten after each epoch.")
