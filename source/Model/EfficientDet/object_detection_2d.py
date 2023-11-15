import omegaconf
import torch
from source.Logger.logger import log_error
from source.LR_Scheduler.on_plateau_policy import OnPlateauLRScheduler
from source.Loss.focalAndSmoothL1_Loss import Focal_SmoothL1_Loss_OD
from source.Model.model_wrapper import ModelWrapper
from omegaconf.dictconfig import DictConfig

class EfficientDet_obj2D_ModelWrapper(ModelWrapper):
    r"""
    # Model Wrapper for 2D Object Detection using EfficientDet
    
    Custom model wrapper class to setup the current process regarding a 2D object detection task
    using EfficientDet as the network architecture.
    """
    def __init__(self, cfg: DictConfig, mode: str) -> None:
        r"""
        Args:
        
        * `cfg (omegaconf.DictConfig)`: 
            * Hydra based configuration dictionary based on the given configuration .yaml file. 
        * `mode (str)`: 
            * The mode of the current process.
        """
        super().__init__(cfg, mode)

    def set_batch(self, batch):
        r"""
        In this method the incoming batch from the dataset is processed. If the batch is wrongly
        used, i.e. by using a wrong key for the dictionary, it will throw an error and terminate.

        Args:

        * `batch (torch.Tensor)`:
            * Incoming batch from the dataset
            * The batch contains the image and the annotation {'img' : img, 'annotation' : annotation}.
        """
        try:
            self.input = (batch['img'].to(torch.device(self.cfg.experiment.on_device)),)
            self.ground_truth = batch['annotation'].to(torch.device(self.cfg.experiment.on_device))
        except KeyError:
            log_error("[ModelWrapper] Unsupported access to current batch format. Check how the batch is made up in the dataset.")
        except omegaconf.errors.ConfigAttributeError:
            log_error("[ModelWrapper] cfg.experiment.on_device in the configuration file is missing.")

    def set_loss_function(self):
        r"""
        Setting the loss function for the 2D object detection task. In this case a mix of the
        FocalLoss for the classification branch and the SmoothL1 loss for the regression part is
        used.

        In this case Focal_SmoothL1_Loss_OD is used from the library.
        """
        self.cls_reg_loss = Focal_SmoothL1_Loss_OD(device=self.cfg.experiment.on_device)
        self.model_arch.anchors = self.model_arch.anchors.to(torch.device(self.cfg.experiment.on_device))

    def compute_loss(self):
        r"""
        Setting `self.final_loss` by computing the loss value. Previously defined functions from
        `set_loss_function()` will be used for that.
        """
        _, r, c = self.prediction
        cls_loss, reg_loss = self.cls_reg_loss(r, c, self.model_arch.anchors, self.ground_truth)
        self.final_loss = cls_loss + reg_loss

    def set_optimizer(self):
        r"""
        Initializing of the optimizer. In this case it is the Adam optimizer with wight decay.
        Required information that needs to be defined in the configuration file is the start point
        of the learning rate.

        Since this information is part of the optimizer the entry should 
        `self.cfg.model_wrapper.optimizer.learning_rate`.
        """
        self.optimizer = torch.optim.AdamW(
            params      =   self.model_arch.parameters(),
            lr          =   self.cfg.model_wrapper.optimizer.learning_rate,
        )

    def set_learning_rate_scheduler(self):
        r""" 
        Initializes the scheduler for the learning rate. In this case the OnPlateauLRScheduler is
        used that also takes into account the validation progress.

        Additionally, a warmup phase was implemented for LRScheduler which is performed for the
        first 25 steps.
        """
        assert self.optimizer is not None, "[ModelWrapper] Optimizer has to be initialized before."
        self.lr_scheduler = OnPlateauLRScheduler(
            optimizer    = self.optimizer,
            init_lr      = self.cfg.model_wrapper.optimizer.learning_rate,
            patience     = 10,
            do_warmup    = True,
            peak_lr      = self.cfg.model_wrapper.optimizer.learning_rate * 10,
            warmup_steps = 50,
            factor       = 0.975
        )

    def update_learning_rate(self, **kwargs):
        r""" 
        This function is called to update the learning rate according to the defined learning rate
        scheduler. Since OnPlateauLRScheduler is used which requires the information about the
        validation loss, the key-value pair `mean_valid_loss` must be given to the method call.

        Since the agent itself does not call this function by default, in used custom agent class
        this method is called after each validation step (after each epoch). 
        """
        self.lr_scheduler.step(kwargs['mean_valid_loss'])
