import torch
import omegaconf

from tqdm.std import tqdm
from source.Model import * # Fetch all model wrappers
from torch.backends import cudnn
from source.Logger.logger import *
from source.Model.model_wrapper import ModelWrapper
from torch.utils.tensorboard import SummaryWriter

cudnn.benchmark = True


class BaseAgent:
    r"""
    # Base Agent
    
    Base class for custom agents. 
    The purpose of the agents is to track, log and make use of custom calls during the training and/or evaluation process.
    """
    def __init__(self, cfg: omegaconf.DictConfig, mode: str) -> None:
        r"""
        Args:

        * `cfg (omegaconf.DictConfig)`: 
            * Hydra based configuration dictionary based on the given configuration .yaml file. 
        * `mode (str)`: 
            * The mode of the current process.
        """
        assert mode.lower() in ['train', 'eval'], \
            "[BaseAgent] Unsupported mode for agent. Only train and eval mode are supported."
        
        self.cfg: omegaconf.DictConfig  = cfg
        self.mode: str = mode

        self._setup_model_wrapper()
        self._setup_logger()

        self.idx_epoch: int = 0
        self.idx_batch: int = 0

        try:
            self.sample_ratio_per_epoch: float = self.cfg.experiment.sample_ratio_per_epoch
            if self.sample_ratio_per_epoch > 1 or self.sample_ratio_per_epoch <= 0:
               log_error("[BaseAgent] sample_ratio_per_epoch must be a value between 0 and 1.")
        except omegaconf.errors.ConfigAttributeError:
            self.sample_ratio_per_epoch: float = 1.0
            log_warning("[BaseAgent] sample_ratio_per_epoch is not set. Training will go over the whole dataset each epoch.")

    def _setup_model_wrapper(self) -> None:
        r""" Setup of custom model wrapper from configuration .yaml file. """
        try:
            model_wrapper_class_str: str = self.cfg.model_wrapper.name
            model_wrapper_class: ModelWrapper = globals()[model_wrapper_class_str]
        except omegaconf.errors.ConfigAttributeError:
            log_error("[BaseAgent] Missing attribute in configuration file cfg.model_wrapper.name.")
        except KeyError:
            log_error("[BaseAgent] ModelWrapper class {} could not be found under source/models/.".format(model_wrapper_class_str))

        log_info("[BaseAgent] Create ModelWrapper {} instance.".format(model_wrapper_class_str))
        self.model_wrapper: ModelWrapper = model_wrapper_class(cfg=self.cfg, mode=self.mode)

    def _setup_logger(self) -> None:
        r""" Setup of logger instances to keep track of progress and visualization. """
        try:
            if self.cfg.agent.use_tensorboard:
                self.tb_log: SummaryWriter = SummaryWriter(log_dir=self.cfg.experiment.summary_dir, comment='FCN8s', flush_secs=1)
                log_info("[BaseAgent] TensorBoard was initialized with log_dir: {}.".format(self.cfg.experiment.summary_dir))
            else:
                raise omegaconf.errors.ConfigAttributeError
        except omegaconf.errors.ConfigAttributeError:
            self.tb_log = None
            log_info("[BaseAgent] TensorBoard was not initialized.")
    
    def _start_training(self) -> None:
        r""" Main training loop in which subcalls for training and validation for an epoch are made. """
        ## Iterate over the number of defined epochs from the configuration file.
        for epoch in range(self.cfg.experiment.num_epochs):
            
            self.idx_epoch = epoch
            self.start_train_valid_epoch()
        
            ## Training part within the loop including abstract function calls that can be used in the custom Agent instance.
            self.model_wrapper._to_train()
            self._train_one_epoch()
            self.end_train_epoch()

            ## Validation part within the loop including abstract function calls that can be used in the custom Agent instance.
            self.model_wrapper._to_eval()
            self._validate()
            self.end_valid_epoch()

            ## Save checkpoint at the end of the training loop.
            try:
                if (epoch % self.cfg.experiment.save_checkpoint_nth_epoch) == 0:
                    self.model_wrapper._save_network(epoch)
            except omegaconf.errors.ConfigAttributeError:
                self.model_wrapper._save_network(epoch)

        self.finalize_epochs()

    def _train_one_epoch(self) -> None:
        r""" 
        Method that processes one epoch for training.
        Besides loading of data it includes the inference, loss computation and parameter update.
        """
        assert self.model_wrapper._training_active(), \
            "[BaseAgent] Model must be in training mode when training process is ongoing!"

        ## The dataloader is separated in batches that is looped over.
        ## It is possible to setup an 'early-stop' by using the sample_ratio_per_epoch term.
        stop_at_sample = int(len(self.model_wrapper.dataloader_training) * self.sample_ratio_per_epoch)
        tqdm_batch = tqdm(self.model_wrapper.dataloader_training, desc="[Training] Epoch-{}/{}".format(self.idx_epoch, self.cfg.experiment.num_epochs), total=stop_at_sample)        
        for i_iter, i_batch in enumerate(tqdm_batch):

            self.idx_batch = i_iter
            self.start_train_batch()

            ## The batch (usually containing input and groundtruth) is provided to the model wrapper.
            self.model_wrapper.set_batch(i_batch)

            ## Run the inference on the given batch.
            self.model_wrapper._forward()

            ## Compute the loss which is defined in the model wrapper itself.
            self.model_wrapper.final_loss = None
            self.model_wrapper.compute_loss()

            assert self.model_wrapper.final_loss is not None, \
                "[BaseAgent] final_loss was not set in model_wrapper.compute_loss()!"
            self.end_train_loss()

            ## Parameters will be optimized as a final step.
            self.model_wrapper._optimize_parameters()
            self.end_train_optim_param()

            if i_iter > stop_at_sample:
                break

    def _validate(self) -> None:
        r""" 
        Method that processes one epoch for validation.
        It is almost equal to the training process beside that not parameters from the model will be updated.
        """
        assert not self.model_wrapper._training_active(), \
            "[BaseAgent] Model must be in validation mode when validation process is ongoing!"

        tqdm_batch = tqdm(self.model_wrapper.dataloader_valid, desc="[Validation] Epoch-{}/{}".format(self.idx_epoch, self.cfg.experiment.num_epochs))
        ## Gradient calculation in validation step is disabled by default.
        with torch.no_grad():
            for i_iter, i_batch in enumerate(tqdm_batch):

                self.idx_batch = i_iter
                self.start_valid_batch()

                ## The batch (usually containing input and groundtruth) is provided to the model wrapper.
                self.model_wrapper.set_batch(i_batch)

                ## Run the inference on the given batch.
                self.model_wrapper._forward()

                ## Compute the loss which is defined in the model wrapper itself.
                self.model_wrapper.final_loss = None
                self.model_wrapper.compute_loss()

                if self.model_wrapper.final_loss is None:
                    log_error("[BaseAgent] final_loss was not set in model_wrapper.compute_loss().")

                self.end_valid_batch()

    def _finalize(self) -> None:
        r""" Final step when training/evaluation process is finished or was aborted. """
        log_info("[BaseAgent] Please wait while finalizing the operation.. Thank you")

        ## Dump Dataset/Dataloader information. 
        if self.model_wrapper.dataset_valid is not None:
            self.model_wrapper.dataset.finalize(self.cfg.experiment.out_dir, self.model_wrapper.dataset.__class__.__name__ + "_dataset_train.csv")
            self.model_wrapper.dataset_valid.finalize(self.cfg.experiment.out_dir, self.model_wrapper.dataset_valid.__class__.__name__ + "_dataset_valid.csv")
        else:
            self.model_wrapper.dataset.finalize(self.cfg.experiment.out_dir, self.model_wrapper.dataset.__class__.__name__ + "_dataset.csv")
        
        if self.tb_log is not None:
            self.tb_log.close()

    def start_train_valid_epoch(self) -> None:
        r""" **Abstract**: This method is called after a new epoch has started in the training/evaluation process. If not overloaded by custom agent it does nothing. """
        pass

    def start_train_batch(self) -> None:
        r""" **Abstract**: This method is called before one batch of the current epoch is going to be processed within in the training process. If not overloaded by custom agent it does nothing. """
        pass

    def end_train_loss(self) -> None:
        r""" **Abstract**: This method is called after the batch was fed to the underlying network and the loss was computed. If not overloaded by custom agent it does nothing. """
        pass

    def end_train_optim_param(self) -> None:
        r""" **Abstract**: This method is called after the optimizer of the ModelWrapper was optimized. If not overloaded by custom agent it does nothing. """
        pass

    def end_train_epoch(self) -> None:
        r""" **Abstract**: This method is called after one epoch was finally processed within the training process. If not overloaded by custom agent it does nothing. """
        pass

    def start_valid_batch(self) -> None:
        r""" **Abstract**: This method is called before one batch of the current epoch is going to be processed in the validation process. If not overloaded by custom agent it does nothing. """
        pass

    def end_valid_batch(self) -> None:
        r""" **Abstract**: This method is called after the batch was fed to the underlying network in the validation process. If not overloaded by custom agent it does nothing. """
        pass

    def end_valid_epoch(self) -> None:
        r""" **Abstract**: This method is called after one epoch was finally processed withing the validation/evaluation process. If not overloaded by custom agent it does nothing. """
        pass

    def finalize_epochs(self) -> None:
        r""" **Abstract**: This method is called when the last epoch was finally processed. If not overloaded by custom agent it does nothing. """
        pass

    def start_eval_batch(self) -> None:
        r""" **Abstract**: This method is called before one batch was fed to the underlying network in the evaluation process. If not overloaded by custom agent it does nothing. """
        pass

    def end_eval_batch(self) -> None:
        r""" **Abstract**: This method is called after the batch was fed to the underlying network in the evaluation process. If not overloaded by custom agent it does nothing. """
        pass
    
    def end_of_evaluation(self) -> None:
        r""" **Abstract**: This method is called when the evaluation process is finished. If not overloaded by custom agent it does nothing. """
        pass
