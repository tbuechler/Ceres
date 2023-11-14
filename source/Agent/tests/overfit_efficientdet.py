import torch
from omegaconf import DictConfig
from source.Utility.tracker import AverageMeter
from source.Agent.base_agent import BaseAgent
from source.Logger.tensorboard import show_scalar, show_scalars, show_image
from source.Utility.ObjectDetection2D.bounding_box2D import visualize
from source.Network.EfficientDet.efficient_det import EfficientDet
from source.Utility.ObjectDetection2D.utils import tensor_to_BBox2D


class Overfit_EfficientDet_obj2D_Agent(BaseAgent):
    r"""
    # Agent: ObjectDetection2D using EfficientDet
    
    Custom agent class to train the EfficientDet model architecture on a 2D ObjectDetection task.
    """
    def __init__(self, cfg: DictConfig, mode: str) -> None:
        r"""        
        Arg:
        
        * `cfg (omegaconf.DictConfig)`: 
            * Hydra based configuration dictionary based on the given configuration .yaml file. 
        * `mode (str)`: 
            * The mode of the current process.
        """
        super().__init__(cfg=cfg, mode=mode)
        
        self.loss_train_tracker: AverageMeter = AverageMeter('Train loss')
        self.loss_valid_tracker: AverageMeter = AverageMeter('Validation loss')

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
        r"""
        This method is called after the loss was computed based on the prediction.
        It will track the result and compute the average over time.
        """
        self.loss_train_tracker.update(self.model_wrapper.final_loss.cpu())

    def end_train_epoch(self) -> None:
        r""" This method is called after one epoch was finally processed within the training process """
        ## Based on the regression and classification output bounding boxes
        ## are created. They will be visualized on top of the input image in
        ## TensorBoard.
        regression_output     = self.model_wrapper.prediction[1].detach().cpu()
        classification_output = self.model_wrapper.prediction[2].detach().cpu()
        bbs =   EfficientDet.decode_output(
                    self.model_wrapper.model_arch.anchors.cpu(),
                    regression_output
                )
        bbs =   tensor_to_BBox2D(
                    decoded_output=bbs, 
                    classification=classification_output, 
                    score_threshold=0.2, 
                    iou_threshold=0.2
                )
        bb_to_show =    visualize(
                            img=self.model_wrapper.input[0][0].detach().cpu(), 
                            boxes=bbs[0], 
                            objectList=self.model_wrapper.dataset.classes,
                            imshow=True, 
                            waitKey=1
                        )
        show_image(
            writer=self.tb_log, tag="Image Train", 
            image_tensor=torch.from_numpy(bb_to_show).permute(2, 0, 1),
            global_step=self.idx_batch
        )

    def end_valid_batch(self) -> None:
        r"""
        This method is called after the loss was computed based on the prediction during validation.
        It will track the result and compute the average over time.
        """
        self.loss_valid_tracker.update(self.model_wrapper.final_loss.cpu())

    def end_valid_epoch(self) -> None:
        r"""
        This method is called after one epoch was finally processed withing the validation/evaluation process.
        It will plot the average loss value for one epoch for the training validation part. Since the OnPlateu-Scheduler
        is used for updating the learning rate, it will also inform the model wrapper about the validation error.
        """
        show_scalars(
            writer=self.tb_log, 
            tag="Computed Loss over epoch", 
            train_loss=self.loss_train_tracker(), 
            valid_loss=self.loss_valid_tracker(),
            global_step=self.idx_epoch
        )

        self.model_wrapper.update_learning_rate(mean_valid_loss=self.loss_valid_tracker())

        self.loss_valid_tracker.reset()
        self.loss_train_tracker.reset()