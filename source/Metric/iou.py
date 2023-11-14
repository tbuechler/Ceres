import torch
import torch.nn as nn
from source.Logger.logger import log_error, log_warning


class IoU(nn.Module):
    """
    Class for calculation of the Intersection-Of-Union.
    """
    def __init__(self, num_classes: int, ignore_val: int=None, smooth: float=1e-5, reduction: str='none') -> None:
        r"""
        Args:
        
        * `num_classes (int)`: 
            * Number of different classes to be considered.
        * `ignore_val (int)`: 
            * Unique value that makes it possible to ignore some values.
        * `smooth (float)`: 
            * Smooth value to avoid zero division if class does not occur in current batch.
        * `reduction (str)`: 
            * Reduction variant. Mean returns mIoU over all classes/batches sum the sum and none returns the class-wise intersection of union.
        """
        super(IoU, self).__init__()
        self.num_classes = num_classes
        self.ignore_val  = ignore_val
        self.smooth      = smooth
        self.reduction   = reduction
        if self.smooth > 1e-1:
            log_warning("[IoU] Smooth value might be too big. This could distort the final result.")
        if self.reduction not in ['mean', 'sum', 'none']:
            log_error("[IoU] Not supported reduction variant: {}.".format(self.reduction))

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r""" Computation of the intersection of union. """
        assert pred_logits.shape[1] == self.num_classes, "Value of channel dimension must be equal to the number of class."
        max_pred = torch.argmax(pred_logits, dim=1)
        assert max_pred.shape == target.shape

        _prediction = max_pred.view(-1)
        _target     = target.view(-1)

        if self.ignore_val is not None:
            _idx2keep   = [_target != self.ignore_val]
            _target     = _target[_idx2keep]
            _prediction = _prediction[_idx2keep]

        _bin_intersection   = torch.bincount(_target[_target == _prediction], minlength=self.num_classes)

        _bin_pred_rest = torch.bincount(_prediction[_target != _prediction], minlength=self.num_classes)
        _bin_target_rest = torch.bincount(_target[_target != _prediction], minlength=self.num_classes)

        _iou = torch.div(_bin_intersection, torch.add(torch.add(_bin_intersection, _bin_pred_rest), torch.add(_bin_target_rest, self.smooth)))

        if self.reduction in 'mean':
            return _iou.mean()
        elif self.reduction in 'sum':
            return _iou.sum()
        else:
            return _iou

