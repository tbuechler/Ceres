from time import sleep
import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    r"""
    Weighted binary cross entropy loss is adopted to deal with the imbalanced problem of boundary detection since coarse boundary is preferred to highlight the boundary region and enhance the features for small objects.

    Example::
        >> pred = torch.randn(4, 1, 128, 128, requires_grad=True)
        >> gt = torch.zeros(4, 128, 128)
        >> gt[:, 5, :] = 1
        >> criterion = BoundaryLoss()
        >> loss = criterion(pred, gt)
        >> print(loss)
        tensor(0.4998) # Just an example    
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, i_pred: torch.Tensor, i_gt: torch.Tensor):
        ## Reshape input to [1, '-1']
        _pred = i_pred.permute(0, 2, 3, 1).contiguous().view(1, -1)
        _gt   = i_gt.view(1, -1)

        ## Get positive (=1) and negative (=0) indices from ground truth image
        pos_idx = (_gt == 1)
        neg_idx = (_gt == 0)

        ## Compute weight vector according to number of positive and negative elements
        _weights = torch.zeros_like(_pred)
        pos_sum = pos_idx.sum()
        neg_sum = neg_idx.sum()
        all_sum = pos_sum + neg_sum
        ## Pos_sum is lower so it must be higher weighted
        _weights[pos_idx] = 1. - (pos_sum / all_sum)
        _weights[neg_idx] = 1. - (neg_sum / all_sum)

        return F.binary_cross_entropy_with_logits(
            input=_pred,
            target=_gt,
            weight=_weights
        )

