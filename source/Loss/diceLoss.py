import torch
import torch.nn as nn
import torch.nn.functional as F
from source.Utility.converter import pytorch_tensor_OneHot
from source.Logger.logger import *

class DiceLoss(nn.Module):
    """
    Dice Loss for multiclass segmentation.
    
    Parameters
    ----------
    num_classes : int
        Number of classes, used for OneHot-Encoding
    ignore_idx : int
        Index in target tensor to be ignored.
    smooth : float
        Smooth value for denominator.
    eps : float
        Smoothing value.
    """
    def __init__(self, num_classes: int=None, ignore_index: int=None, smooth: float=1e-4, eps: float=1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.smooth      = smooth
        self.eps         = eps
        self.ignore_idx  = ignore_index
        log_info("[DiceLoss] DiceLoss uses softmax to enforce the sum of the channel dimension to be one. Make sure that F.softmax(x, dim=1) is correct.")
        
    def forward(self, logits: torch.Tensor, targetLabel: torch.Tensor) -> torch.Tensor:
        """
        Calculates dice loss.
        Parameters
        ----------
        logits : torch.Tensor
            Direct output from the model (logits) with shape of [batch, channel, height, width].
        targetLabel : torch.Tensor
            Ground-truth class index tensor with shape of [batch, height, width].
        Returns
        -------
        losses_and_metrics : tensor
            Overall dice loss
        """
        ## Use a differentiable function where the sum of the channel dimension is equal to one, i.e. F.softmax(...)
        ## Do NOT use argmax, since this function is not differentiable

        logits = F.softmax(logits, dim=1)
        ## Indices which must be ignored, are set to zero in this One-Hot configuration.
        encoded_target    = pytorch_tensor_OneHot(targetLabel, ignore_idx=self.ignore_idx, num_classes=self.num_classes)    

        numerator   = 2 * torch.mul(encoded_target, logits).sum((2, 3)) + self.smooth
        denominator = logits.sum((2, 3)) + encoded_target.sum((2, 3)) + self.smooth + self.eps

        return torch.mean((1. - numerator / denominator), dim=(0, 1))

