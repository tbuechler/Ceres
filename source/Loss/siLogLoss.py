import torch
import torch.nn as nn


class SILogLoss(nn.Module):
    def __init__(self, ratio_alpha, ratio_beta):
        """ Scaled version of the Scale-Invariant loss (SI) proposed in https://arxiv.org/abs/1406.2283"""
        super(SILogLoss, self).__init__()
        self.ratio1 = ratio_alpha
        self.ratio2 = ratio_beta

    def forward(self, prediction: torch.Tensor, groundtruth: torch.Tensor, mask=None, interpolate=True):
        if interpolate:
            prediction = nn.functional.interpolate(prediction, groundtruth[-2:], mode='bilinear', align_corners=True)
        
        if mask is not None:
            prediction = prediction[mask]
            groundtruth = groundtruth[mask]
        
        log_diff = torch.log(prediction * self.ratio1) - \
                   torch.log(groundtruth * self.ratio1)
        Dg = torch.var(log_diff) + self.ratio2 * torch.pow(torch.mean(log_diff), 2)
        return self.ratio1 * torch.sqrt(Dg)
