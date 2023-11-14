import torch
import omegaconf
from source.Logger.logger import log_error
from source.LR_Scheduler.on_plateau_policy import OnPlateauLRScheduler
from source.Loss.boundaryLoss import BoundaryLoss
from source.Loss.crossEntropy import CrossEntropyLoss, CrossEntropy_BoundaryAwarenessLoss
from source.Model.model_wrapper import ModelWrapper
from source.Dataloader.Utility.converter import labImg2EdgeTorch
from omegaconf.dictconfig import DictConfig


class PIDNet_Semseg_ModelWrapper(ModelWrapper):
    r"""
    # Model Wrapper for Semantic Segmentation task using PIDNet.
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

    def set_batch(self, batch: torch.Tensor) -> None:
        r"""
        In this method the incoming batch from the dataset is processed. If the batch is wrongly used, i.e. by using a wrong key for the dictionary, it will log an error and terminate the process.

        Args:

        * `batch (dict(torch.Tensor))`:
            * Incoming batch from the dataset
            * The batch contains the image and the pixelwise labeled image {'img_left' : img, 'label' : label}.
        """
        try:
            edge = torch.stack([labImg2EdgeTorch(batch['label'][i]) for i in range(batch['label'].shape[0])]).type(torch.float32)
            self.input = (batch['img_left'].to(torch.device(self.cfg.experiment.on_device)),)
            self.ground_truth = (batch['label'].to(torch.device(self.cfg.experiment.on_device)), edge.to(torch.device(self.cfg.experiment.on_device)))
        except KeyError:
            log_error("[ModelWrapper] Unsupported access to current batch format. Check how the batch is made up in the dataset.")
        except omegaconf.errors.ConfigAttributeError:
            log_error("[ModelWrapper] cfg.experiment.on_device in the configuration file is missing.")

    def set_loss_function(self):
        r"""
        Setting the loss function for the semantic segmentation task. In this case three different loss functions are being used for training:

        1. Boundary Loss used at the D output which is basically a weighted Cross Entropy loss to overcome the imbalanced problem at boundaries.
        2. Cross Entropy loss for the output from the P and I layer.
        3. Cross Entropy loss with awareness of boundary used at I and D layer.
        """
        self.loss_l1 = BoundaryLoss()
        self.loss_l0_l2 = CrossEntropyLoss(
            weights=self.dataset.class_weights.to(torch.device(self.cfg.experiment.on_device)), 
            ignore_index=self.dataset.ignore_label, 
            reduction='mean'
        )
        self.loss_l3 = CrossEntropy_BoundaryAwarenessLoss(
            boundary_threshold=0.8, 
            class_ignore_label=255, 
            weights=self.dataset.class_weights.to(torch.device(self.cfg.experiment.on_device)), 
            reduction='mean'
        )

    def compute_loss(self):
        r"""
        Setting `self.final_loss` by computing the loss value. Previously defined functions from `set_loss_function()` will be used for that.
        
        $$ Loss = \lambda_0 l_\lambda + \lambda_1 l_1 + \lambda_2 l_2 + \lambda_3 l_3$$
        where
        $$\lambda_3 = -\sum_{i, c}\{1:b_i>\tau\}(s_{i, c}logs_{i, c})$$
        with $\lambda_0=0.4, \lambda_1 = 20, \lambda_2 = 1, \lambda_3 = 1$ and $\tau = 0.8$.

        Since during evaluation only one output path is available, only the second loss will be computed.
        """
        if self.model_arch.training:
            p_out, i_out, d_out = self.prediction
            loss0 = self.loss_l0_l2(p_out, self.ground_truth[0])     * 0.4
            loss1 = self.loss_l1(d_out, self.ground_truth[1])        * 20.
            loss2 = self.loss_l0_l2(i_out, self.ground_truth[0])     * 1.
            self.final_loss = loss0 + loss1 + loss2
            loss3 = self.loss_l3(d_out, i_out, self.ground_truth[0]) * 1.
            if not torch.isnan(loss3):
                self.final_loss += loss3
        else:
            i_out = self.prediction
            self.final_loss = self.loss_l0_l2(i_out, self.ground_truth[0])

    def set_optimizer(self):
        r"""
        Initializing of the optimizer. In this case the Adam optimizer with wight decay is being used. Required information that needs to be defined in the configuration file is the start point of the learning rate.

        Since this information is part of the optimizer the entry should `self.cfg.model_wrapper.optimizer.learning_rate`.
        """
        self.optimizer = torch.optim.AdamW(
            params      =   self.model_arch.parameters(),
            lr          =   self.cfg.model_wrapper.optimizer.learning_rate,
        )

    def set_learning_rate_scheduler(self):
        r""" 
        Initializes the scheduler for the learning rate. In this case the OnPlateauLRScheduler is used that also takes into account the validation progress.

        Additionally, a warmup phase was implemented for LRScheduler which is performed for the first 25 steps.
        """
        assert self.optimizer is not None, "[ModelWrapper] Optimizer has to be initialized before."
        self.lr_scheduler = OnPlateauLRScheduler(
            optimizer=self.optimizer,
            init_lr=self.cfg.model_wrapper.optimizer.learning_rate,
            patience=10,
            do_warmup=True,
            peak_lr=self.cfg.model_wrapper.optimizer.learning_rate * 10,
            warmup_steps=100,
            factor=0.975
        )

    def update_learning_rate(self, **kwargs):
        r""" 
        This function is called to update the learning rate according to the defined learning rate scheduler. Since OnPlateauLRScheduler is used which requires the information about the validation loss, the key-value pair `mean_valid_loss` must be given to the method call.

        Since the agent itself does not call this function by default, in used custom agent class this method is called after each validation step (after each epoch). 
        """
        self.lr_scheduler.step(kwargs['mean_valid_loss'])

