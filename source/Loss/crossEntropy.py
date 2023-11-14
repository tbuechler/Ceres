import torch
import torch.nn.functional as F
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    r"""
    Wrapper class for the Cross Entropy loss.

    Args::
        weights (torch.Tensor): Weight tensor for all classes.
        ignore_index (int): Index to be ignored in computation.
        reduction (str): Reduction method used for the final loss.

    Example::
        >> cls_pred = torch.randn(4, 19, 128, 128, requires_grad=True)
        >> gt = torch.zeros(4, 128, 128, dtype=torch.long).random_(19)
        >> criterion = CrossEntropyLoss(ignore_index=255)
        >> loss = criterion(cls_pred, gt)
        print(loss)
    """
    def __init__(self, weights: torch.Tensor=None, ignore_index: int=255, reduction: str='mean'):
        super().__init__()
        self.ignore_idx = ignore_index
        self.ce_criterion = nn.CrossEntropyLoss(
            weight=weights,
            ignore_index= ignore_index,
            reduction=reduction
        ) 

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_criterion(inputs, targets)


class CrossEntropy_BoundaryAwarenessLoss(nn.Module):
    r"""
    Boundary-awareness Cross Entropy loss method inspired by 'Gated shape cnns for semantic segmentation'. It uses the input of boundary information to coordinate semantic segmentation.
    
    Args::
        boundary_threshold: Threshold to handle value as positive or negative boundary.
        ignore_index (int): Index to be ignored in computation.
        weights (torch.Tensor): Weight tensor for all classes.
        reduction (str): Reduction method used for the final loss.

    Example::
        >> bnd_pred = torch.randn(4, 1, 128, 128, requires_grad=True)
        >> cls_pred = torch.randn(4, 19, 128, 128, requires_grad=True)
        >> gt = torch.zeros(4, 128, 128, dtype=torch.long).random_(19)
        >> criterion = CrossEntropy_BoundaryAwarenessLoss(0.8, 255)
        >> loss = criterion(bnd_pred, cls_pred, gt)
        >> print(loss)
        tensor(0.4998) # Just an example 
    """
    def __init__(self,
        boundary_threshold: float = 0.8,
        class_ignore_label: int=-1,
        weights:            torch.Tensor=None,
        reduction:          str='mean'
    ) -> None:
        super().__init__()
        self.bd_threshold = boundary_threshold
        self.cls_ignore   = class_ignore_label

        self.ce_criterion = CrossEntropyLoss(
            weights=weights,
            ignore_index=self.cls_ignore,
            reduction=reduction
        )
    
    def forward(self, boundary_pred, class_pred, class_gt):
        ## Adjust label vector according to boundary prediction
        ## If boundary value is above threshold use actual class label, otherwise use ignore value
        tmp = torch.ones_like(class_gt) * self.cls_ignore
        _gt = torch.where(
            torch.sigmoid(boundary_pred[:,0,:,:]) > self.bd_threshold,
            class_gt, tmp
        )
        return self.ce_criterion(class_pred, _gt)


class OhemCELoss(nn.Module):
    """
    Cross Entry loss with Online Hard Example Mining. 
    
    Parameters
    ----------
    min_kept : float
        
    threshold : float
        
    weights : torch.FloatTensor
        Weight tensor for all classes in the dataset.

    ignore_index : int
        Index which has to be ignored in the loss calculation.

    reduction : str
        How the separate losses are reduced in the end.
    """
    def __init__(self, weights: torch.FloatTensor=None, min_kept: int=0, threshold: float=0.6, ignore_index: int=None, reduction: str='none'):
        super().__init__()
        self.min_kept    = min_kept
        self.threshold   = threshold

        # It makes no difference if parameter weight is specified as None or not passed at all
        if ignore_index is not None:
            self.criteria = nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index, reduction=reduction)
        else:
            self.criteria = nn.CrossEntropyLoss(weight=weights, reduction=reduction)


    def forward(self, prediction_logits: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates Cross Entry loss with Online Hard Example Mining.

        Parameters
        ----------
        prediction_logits : torch.Tensor
            Direct output from the model (logits) with shape of [batch, channel, height, width].
        target_tensor : torch.Tensor
            Ground-truth class index tensor with shape of [batch, height, width].

        Returns
        -------
        losses_and_metrics : tensor
            Mean of channelwise OhemCELoss
        """
        loss    = self.criteria(prediction_logits, target_tensor).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.min_kept] > self.threshold:
            loss = loss[loss>self.threshold]
        else:
            loss = loss[:self.min_kept]
        return torch.mean(loss)

class BinaryCELoss(nn.Module):
    """
    Binary Cross Entropy loss for binary classification. 
    """
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights

    def forward(self, prediction_logits: torch.Tensor, target_tensor: torch.Tensor) -> torch.Tensor:
        """
        Calculates Binary Cross Entropy.

        Parameters
        ----------
        prediction_logits : torch.Tensor
            Direct output from the model (logits) with shape of [batch, channel, height, width].
        target_tensor : torch.Tensor
            Ground-truth class index tensor with shape of [batch, height, width].

        Returns
        -------
        losses_and_metrics : tensor
            Mean of channelwise OhemCELoss
        """
        return F.binary_cross_entropy_with_logits(prediction_logits, target_tensor, weight=self.weights)


