import torch
import omegaconf

from source.Network import * # Fetch all possible network architectures
from source.Dataloader import * # Fetch all datasets
from source.Logger.logger import *
from source.Utility.system import *
from source.Network.utils import get_network_parameter_number
from source.Dataloader.Utility.dataloader import create_dataloader


class ModelWrapper:
    r"""
    # Model Wrapper

    Base class for a custom model wrapper.
    """
    def __init__(self, cfg: omegaconf.DictConfig, mode: str) -> None:
        r"""
        Args:
        
        * `cfg (omegaconf.DictConfig)`: 
            * Hydra based configuration dictionary based on the given configuration .yaml file. 
        * `mode (str)`: 
            * The mode of the current process.
        """
        super().__init__()
        self.mode       = mode
        self.cfg        = cfg

        ## Member for automatic data exchange with agent etc.
        self.input          = None  # Input is set after each call of set_batch
        self.ground_truth   = None  # GroundTruth is set after each call of set_batch
        self.prediction     = None  # Prediction is set after each forward call of the network
        self.final_loss     = None  # FinalLoss is set after the loss was computed

        ## Member that hold the whole architecture.
        self.lr_scheduler   = None
        self.optimizer      = None
        self.model_arch     = None

        self._setup()

    def _setup(self) -> None:
        r"""
        Sets up the current model wrapper including network architecture, dataloader, optimizer, etc.
        """
        try:
            network_class = globals()[self.cfg.network.name]
        except omegaconf.errors.ConfigAttributeError:
            log_error("[ModelWrapper] Missing attribute in configuration file cfg.network.name.")
        except KeyError:
            log_error("[ModelWrapper] ModelWrapper class {} could not be found under source/models/.".format(self.cfg.network.name))
        try:
            self.model_arch = network_class(self.cfg)
        except omegaconf.errors.ConfigAttributeError:
            log_error("[ModelWrapper] Missing attribute in configuration file cfg.experiment.on_device.")
        log_info("[ModelWrapper] Initialized {} as network architecture.".format(self.cfg.network.name))

        try:
            self._load_network(path_to_pth_file=self.cfg.experiment.preload_checkpoint)
        except omegaconf.errors.ConfigAttributeError:
            if self.mode.lower() in ["eval"]:
                log_error("[ModelWrapper] Preloading checkpoint is mandatory in eval mode. Entry cfg.experiment.preload_checkpoint was not found.")
            else:
                log_warning("[ModelWrapper] No checkpoint was preloaded. Fresh network architecture will be initialized.")
        self.model_arch = network_class(self.cfg).to(torch.device(self.cfg.experiment.on_device))

        self._print_network()
        self._setup_dataloader()
        self.set_loss_function()

        self.set_optimizer()
        assert self.optimizer is not None, "[ModelWrapper] self.optimizer was not initialized yet what is necessary for set_learning_rate_scheduler()."
        try:
            self._load_optimizer(path_to_pth_file=self.cfg.experiment.preload_optimizer)
        except omegaconf.errors.ConfigAttributeError:
            if self.mode.lower() in ["train"]:
                log_warning("[ModelWrapper] No checkpoint was preloaded. Fresh optimizer will be initialized.")

        self.set_learning_rate_scheduler()

    def _setup_dataloader(self) -> None:
        r""" 
        Setup Dataset and Dataloader defined in the configuration file. Regardless of the mode a dataset and dataloader for training and validation is created. 
        
        After this part was called dataset and dataloader are initialized and accessible via
        
        * self.dataset
        * self.dataset_valid (optional)
        * self.dataloader_train
        * self.dataloader_valid
        """
        try:
            use_multidataset = self.cfg.multi_dataset is not None
        except omegaconf.errors.ConfigAttributeError:
            use_multidataset = False

        if not use_multidataset:
            try:
                dataset_class_str = self.cfg.dataset.name
                dataset_class = globals()[dataset_class_str]        
                log_info("[ModelWrapper] Create instance of dataset {}.".format(dataset_class_str))
                self.dataset          = dataset_class(cfg=self.cfg.dataset)
            except omegaconf.errors.ConfigAttributeError:
                log_error("[ModelWrapper] Missing attribute in configuration file cfg.dataset.name.")
            except KeyError:
                log_error("[ModelWrapper] Dataset class {} could not be found under source/datasets/.".format(dataset_class_str))
        else:
            try:
                multi_dataset_class_str = self.cfg.multi_dataset.name
                multi_dataset_class = globals()[multi_dataset_class_str]        
                log_info("[ModelWrapper] Create instance of multi dataset {}.".format(multi_dataset_class_str))
                
                ds = {} 
                try:
                    for dataset_class_str in self.cfg.multi_dataset.datasets:
                        try:
                            dataset_class = globals()[dataset_class_str]
                        except KeyError:
                            log_error("[ModelWrapper/MultiDataHandler] Dataset class {} could not be found under source/datasets/.".format(dataset_class_str))
                        with omegaconf.open_dict(self.cfg):
                            self.cfg.multi_dataset[dataset_class_str].name = dataset_class_str
                        ds[dataset_class_str] = dataset_class(self.cfg.multi_dataset[dataset_class_str])
                except omegaconf.errors.ConfigAttributeError:
                    log_error("[ModelWrapper/MultiDataHandler] Missing attribute to specify all datasets cfg.datasets.")
                self.dataset          = multi_dataset_class(cfg=self.cfg.multi_dataset, datasets=ds)
            except omegaconf.errors.ConfigAttributeError:
                log_error("[ModelWrapper] Missing attribute in configuration file cfg.multi_dataset.name.")
            except KeyError:
                log_error("[ModelWrapper] Multi Dataset class {} could not be found under source/datasets/.".format(dataset_class_str))

        self.dataset_valid = None
        if self.mode in ['train']:
            try:
                dataset_valid_class_str = self.cfg.dataset_valid.name
                dataset_valid_class     = globals()[dataset_valid_class_str]
                log_info("[BaseAgent] Create instance of separate validation dataset {}.".format(dataset_valid_class_str))
                self.dataset_valid      = dataset_valid_class(cfg=self.cfg.dataset_valid)
            except omegaconf.errors.ConfigAttributeError:
                log_info("[BaseAgent] No different instance created for validation dataset. Try to split up the only one dataset available...")

        else:
            if self.mode.lower() in ["eval"]:
                with omegaconf.open_dict(self.cfg):
                    self.cfg.dataset.validation_ratio = 1

        if self.dataset_valid is None:
            self.dataset.split_train_valid()
        else:
            self.dataset.indices_train = []
            self.dataset.indices_valid = []
        
        log_info("[BaseAgent] Create instances of training and validation dataloader.".format(dataset_class_str))
        self.dataloader_training, self.dataloader_valid = create_dataloader(
            cfg=self.cfg, ds1=self.dataset, ds2=self.dataset_valid, use_multidataset=use_multidataset
        )

    def _forward(self) -> None:
        r"""
        The input, packed as a tuple, is going to be collapsed and fed into the network.
        """
        self.prediction = self.model_arch(*self.input)

    def _optimize_parameters(self) -> None:
        r"""
        Computation of loss, gradients and update of network weights.
        """
        assert self.optimizer is not None, "self.optimizer was not initialized yet."
        assert self.final_loss is not None, "The final computed loss is None."
        assert self.final_loss.requires_grad, "The final computed loss is not inferable." 

        ## If `self.final_loss` is a nan value it must be handled differently.
        ## The optimizer is not called but the nan value cannot be ignored. 
        ## Otherwise the memory allocation is going to be accumulated. It 
        ## needs to be delete manually.
        if torch.isnan(self.final_loss):
            del self.final_loss

        else:
            self.optimizer.zero_grad()
            self.final_loss.backward()
            self.optimizer.step()     

    def _save_network(self, epoch) -> None:
        r""" 
        Method to save the current state of the process including a checkpoint of the network architecture. 
        """
        assert self.input is not None, "[ModelWrapper] self.input must be set in order to trace a model."
        assert not self.model_arch.training, "[ModelWrapper] Expected model in eval mode when saving it."        
        self.model_arch.cpu()        

        if self.cfg.experiment.overwrite_checkpoint:
            dir_name = "{}".format(self.cfg.experiment.name)
        else:
            dir_name = "{}_{}_e{}".format(self.cfg.experiment.name, get_timestamp(), epoch)

        make_dirs([
            os.path.join(self.cfg.experiment.checkpoint_dir, dir_name),
            os.path.join(self.cfg.experiment.checkpoint_dir, dir_name, 'subs')
        ])

        state = {
            'epoch':        epoch,
            'state_dict':   self.model_arch.state_dict(),
            'optimizer':    self.optimizer.state_dict()
        }

        torch.save(
            state, 
            os.path.join(
                os.path.join(self.cfg.experiment.checkpoint_dir, dir_name),
                (dir_name + '_' + self.model_arch.__class__.__name__ + '.pth')
            )
        )

        ## Exporting the network requires a dummy input fro tracing. For this,
        ## it just takes the last used input. If the export was not implement, 
        ## it will just do nothing.
        input_cpu = tuple([x.cpu() for x in self.input])
        self.model_arch.export(
            input_cpu, 
            os.path.join(self.cfg.experiment.checkpoint_dir, dir_name)
        )

        self.model_arch.to(torch.device(self.cfg.experiment.on_device))

    def _load_network(self, path_to_pth_file: str) -> None:
        r"""
        Loads a pretrained network to the process. This method is called in the beginning of each process and will basically do nothing if no model is given.
        """
        try:
            checkpoint = torch.load(f=path_to_pth_file, map_location=torch.device('cpu'))
        except OSError:
            log_error("[ModelWrapper] Not able to load checkpoint from {}.".format(self.cfg.experiment.preload_checkpoint))

        try:
            self.model_arch.load_state_dict(checkpoint['state_dict'])
        except KeyError as e:
            log_error("[ModelWrapper] Key 'state_dict' cannot be found in .pht file: {}.".format(e))

    def _load_optimizer(self, path_to_pth_file: str) -> None:
        r"""
        Loads a saved state of an optimizer that has been used in previous runs.
        """
        try:
            checkpoint = torch.load(f=path_to_pth_file, map_location=torch.device('cpu'))
        except OSError:
            log_error("[ModelWrapper] Not able to load checkpoint from {}.".format(self.cfg.experiment.preload_checkpoint))

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        except KeyError:
            log_warning("[ModelWrapper] Key 'optimizer' cannot be found in .pht file. The optimizer will be reinitialized with fresh parameters.")

    def _print_network(self) -> None:
        r"""
        Prints the total number of trainable parameters in the utilized network architecture.
        """
        assert self.model_arch is not None, "Model architecture was not initialized yet."
        num_params = get_network_parameter_number(self.model_arch)
        log_info('[ModelWrapper] Network {0} Total number of parameters : {1:.3f} M'.format(self.cfg.network.name, num_params / 1e6))

    def _training_active(self) -> None:
        r""" 
        Returns true if the underlying network is in training mode, else False. 
        """
        return self.model_arch.training

    def _to_train(self) -> None:
        r""" Converts the underlying network into training mode. """
        self.model_arch.train()

    def _to_eval(self) -> None:
        r""" Converts the underlying network into evaluation mode. """
        self.model_arch.eval()
        
    def get_lr(self) -> None:
        r""" Returns the current learning rate from the scheduler. """
        return self.lr_scheduler.get_lr()
        
    def update_learning_rate(self, **kwargs) -> None:
        r"""
        **Abstract**: Updates the learning rate based on a predefined approach. This method is **NOT** called from the BaseAgent automatically. This method must be triggered manually within the custom agent using the available abstract methods.
        """
        if self.mode.lower() in ["eval"]:
            pass
        else:
            raise NotImplementedError

    def set_batch(self, batch) -> None:
        r"""
        **Abstract**: Processes the batch coming from the dataset and passed from the agent and sets the internal input member accordingly. This needs to be implemented by the user, since the structure of a batch from different datasets can be different.
        """
        raise NotImplementedError

    def set_loss_function(self) -> None:
        r"""
        **Abstract**: Method to initialize the functions for computing the final loss value. 
        """
        raise NotImplementedError

    def compute_loss(self) -> None:
        r""" 
        **Abstract**: Method to actually compute the loss value using the previously initialized loss functions. After the call of this method `self.final_loss` must be set. 
        """
        raise NotImplementedError

    def set_optimizer(self):
        r""" 
        **Abstract**: Construction of an Optimizer. 
        """
        if self.mode.lower() in ["eval"]:
            pass
        else:
            raise NotImplementedError

    def set_learning_rate_scheduler(self):
        r"""
        **Abstract**: Construction of a LR Scheduler.
        """
        if self.mode.lower() in ["eval"]:
            pass
        else:
            raise NotImplementedError

