import torch
from omegaconf import DictConfig
from source.Utility.tracker import AverageMeter
from source.Agent.base_agent import BaseAgent
from source.Logger.tensorboard import show_scalar, show_scalars, show_image

class Overfit_PIDNet_SemSeg_Agent(BaseAgent):
    r"""
    # Agent for Semantic Segmentation using PIDNet

    Custom agent class to train the PIDNet model architecture on a semantic segmentation task.
    """
    def __init__(self, cfg: DictConfig, mode: str) -> None:
        r"""
        Args:
        
        * `cfg (omegaconf.DictConfig)`: 
            * Configuration dictionary created by Hydra based on the given configuration .yaml file
        * `mode (str)`: 
            * The mode of the current process.
        """
        super().__init__(cfg=cfg, mode=mode)
        
        self.loss_train_tracker = AverageMeter('Train loss')
        self.loss_valid_tracker = AverageMeter('Validation loss')

    def start_train_valid_epoch(self) -> None:
        r""" This method is called after a new epoch has started in the training/evaluation process. """
        ## Plot the value of the learning rate using TensorBoard.
        show_scalar(
            writer=self.tb_log, 
            tag="Learning Rate", 
            data=self.model_wrapper.get_lr(),
            global_step=self.idx_epoch
        )

    def end_train_loss(self) -> None:
        r""" This method is called after the batch was fed to the underlying network and the loss was computed. """
        self.loss_train_tracker.update(self.model_wrapper.final_loss.cpu())

        ## Every 100th iteration the prediction output during training is
        ## visualized in Tensorboard along with the ground truth data.
        if self.idx_batch % 100 == 0:
            _out =  torch.argmax(
                        self.model_wrapper.prediction[1].detach().cpu(), 
                        dim=1
                    )
            show_image(
                writer=self.tb_log, 
                tag="Image Train", 
                image_tensor=self.model_wrapper.input[0][0].detach().cpu(),
                global_step=self.idx_batch
            )
            show_image(
                writer=self.tb_log, 
                tag="Prediction Train", 
                image_tensor=self.model_wrapper.dataset.labelTensor2colorTensor(_out[0]), 
                global_step=self.idx_batch
            )
            show_image(
                writer=self.tb_log, 
                tag="GroundTruth Train", 
                image_tensor=self.model_wrapper.dataset.labelTensor2colorTensor(self.model_wrapper.ground_truth[0][0].detach().cpu()),
                global_step=self.idx_batch
            )

    def end_valid_batch(self) -> None:
        r""" This method is called after the batch was fed to the underlying network in the validation process. """
        self.loss_valid_tracker.update(self.model_wrapper.final_loss.cpu())

        ## Every 100th iteration the prediction output during validation is
        ## visualized in Tensorboard along with the ground truth data.
        if self.idx_batch % 100 == 0:
            _out =  torch.argmax(
                        self.model_wrapper.prediction.detach().cpu(), 
                        dim=1
                    )
            show_image(
                writer=self.tb_log, 
                tag="Image", 
                image_tensor=self.model_wrapper.input[0][0].detach().cpu(),
                global_step=self.idx_batch
            )
            show_image(
                writer=self.tb_log, 
                tag="Prediction", 
                image_tensor=self.model_wrapper.dataset.labelTensor2colorTensor(_out[0]), 
                global_step=self.idx_batch
            )
            show_image(
                writer=self.tb_log, 
                tag="GroundTruth", 
                image_tensor=self.model_wrapper.dataset.labelTensor2colorTensor(self.model_wrapper.ground_truth[0][0].detach().cpu()),
                global_step=self.idx_batch
            )


    def end_valid_epoch(self) -> None:
        r""" This method is called after one epoch was finally processed withing the validation/evaluation process. """
        ## Plot of training and validation loss of current epoch in
        ## Tensorboard.
        show_scalars(
            writer=self.tb_log, 
            tag="Computed Loss over epoch", 
            train_loss=self.loss_train_tracker(), 
            valid_loss=self.loss_valid_tracker(),
            global_step=self.idx_epoch
        )

        ## Trigger update of Learning Rate. Since OnPlateau-Policy is used
        ## the call was moved to the agent where the validartion loss is
        ## known. 
        self.model_wrapper.update_learning_rate(mean_valid_loss=self.loss_valid_tracker())

        self.loss_valid_tracker.reset()
        self.loss_train_tracker.reset()
