import torch
import omegaconf

from omegaconf import DictConfig
from source.Logger.logger import log_error
from source.Dataloader.datahandler import DataHandler


def ds2dl(cfg: DictConfig, ds: DataHandler, indices: int=None, train: bool=False):
    r"""
    Creates a DataLoader instance from a dataset object.

    Args:
    
    * `cfg (omegaconf.DictConfig)`:
        * Configuration items for the dataset [subset of the whole configuration object].
    * `ds (DataHandler)`: 
        * Instantiated dataset object.
    * `indices (list[int])`: 
        * If the dataset was splitted the corresponding list of indices for the DataLoader must be
        given as well. No effect if two different datasets are used.
    * `train (bool)`: 
        * Indicates whether the DataLoader is used for training oder evaluation/validation. This is
         unfortunately necessary to catch missing configuration entries, since it is not mandatory
         to define dataset_valid but dataset needs to be defined.

    Returns:

    * dl (DataLoader): 
        * Created dataloader instance. 
    """
    ##
    ## This part of code is meant to select the correct batch size.
    ##
    ## The following combinations can happen:
    ##
    ## 1. The dataloader is meant for training. Then it is expected to define `batch_size` in your
    ## configuration file. Otherwise it will throw an error and terminate.
    ## 2. The dataloader is meant for validation/evaluation and the dataset used in this call was
    ## split before, so the dataset is used for training **and** validation. This means that
    ## `indices` is not empty. In this case it will try to get the batch size by trying to access
    ## `cfg.batch_size_valid` (Because `batch_size` should be used for training if Dataset is used
    ## for training and validation). IF this entry was not defined in the configuration file, Hydra
    ## will throw an error and the used batch size is set to 1.
    ## 3. The dataloader is meant for validation/evaluation and the dataset used in this call is a
    ## separate instance compared to the dataset used for training - so no split has been performed
    ## before. This leads to `indices` being `None`. If this is the case, it will uses `batch_size`
    ## defined in your `dataset_validation` part of your configuration file. If this entry was not
    ## defined for your validation set, it will set the batch size to 1.
    try:
        batch_size = cfg.batch_size 
        if indices is not None and not train:
            batch_size = cfg.batch_size_valid
    except omegaconf.errors.ConfigAttributeError:
        if train:
            log_error("[DataLoader] No batch_size defined in config file for training set {}.".format(ds.__class__.__name__))
        else:
            batch_size = 1

    ## To enable multi-process dataloading define num_worker in your configuration dataset section.
    ## By default this value is set to four times of your number of GPUs in your system. 
    try:
        num_worker = cfg.num_worker
    except omegaconf.errors.ConfigAttributeError:
        num_worker = 1
            
    sampler = None
    if indices is not None:
        sampler = torch.utils.data.SubsetRandomSampler(indices)
        shuffle = False
    else:
        shuffle = True

    ## If you have implemented a custom collate_fn in your dataset (custom DataHandler), it will be
    ## used in this case. If not, the default process by PyTorch is being used.
    try:
        collate_fn = ds.collate_fn
    except AttributeError:
        collate_fn = None

    ## Definition of return value for each dataloader instance.
    dl = torch.utils.data.DataLoader(
        dataset         = ds,
        batch_size      = batch_size,
        pin_memory      = True,
        drop_last       = True,
        shuffle         = shuffle,
        num_workers     = num_worker,
        sampler         = sampler,
        collate_fn      = collate_fn
    )
    return dl
    
def create_dataloader(cfg: DictConfig, ds1: DataHandler, ds2: DataHandler=None, use_multidataset: bool=False):
    r"""
    Method to call to create dataloader instances from training and validation dataset.

    Args:

    * `cfg (omegaconf.DictConfig)`:
        * Configuration instance for this experiment (no subset).
    * `ds1 (DataHandler)`: 
        * Instantiated first dataset object. Either used for training and validation or only for training.
    * `ds2 (DataHandler)`: 
        * Instantiated second dataset object. If defined, it is used for validation.
    * `use_multidataset (bool)`:
        * Indicates whether a multi dataset is used in this run.
    """
    dl1 = None
    dl2 = None

    ## Distinguish between two cases:
    ##
    ## 1. The second dataset `ds2` is not given and thus `None`. In this case `ds1` must be
    ## splitted before and two dataloader instances are created with the same dataset instance but
    ## considering the indices lists.
    ## 2. The second dataset `ds2` is given. In this case two dataloader instaces are created by
    ## using `ds1` for the training set and `ds2` for the validation set.
    ds_str = "dataset" if not use_multidataset else "multi_dataset"
    if ds2 is None:
        dl1 = ds2dl(cfg=cfg[ds_str], ds=ds1, indices=ds1.indices_train, train=True)
        dl2 = ds2dl(cfg=cfg[ds_str], ds=ds1, indices=ds1.indices_valid, train=False)
    else:
        dl1 = ds2dl(cfg=cfg[ds_str],        ds=ds1, train=True)
        dl2 = ds2dl(cfg=cfg.dataset_valid,  ds=ds2, train=False)
    return dl1, dl2
