import torch
import torch.nn as nn

from source.Loss.l1l2_loss import SmoothL1_Loss
from source.Loss.Common.misc_ops import warp_image_by_disparity


class PhotometricLoss(nn.Module):
    r"""
    Photometric loss for multi-camera situations.

    Penalizes sharp disparity/depth transition in the absence of edges in the reference image, 
    i.e. disparity map and left stereo image, according to

    $$ L_{smooth} = \frac{1}{N}\sum_{j=1}^N |\delta_x d'_j| \cdot e^{-||\delta_x I^l_j||} + 
                    \frac{1}{N}\sum_{j=1}^N |\delta_y d'_j| \cdot e^{-||\delta_y I^l_j||}$$.
    """
    def __init__(self, beta: float=1.0, reduction: str=None) -> None:
        r"""
        Args:

        * `beta (float)`: Threshold at which to change between L1 and L2.
        * `reduction (str)`:
            * reduction: 'none' | 'mean' | 'sum'
                * 'none': No reduction will be applied to the output.
                * 'mean': The output will be averaged.
                * 'sum': The output will be summed.
        """
        super().__init__()
        self.smooth_l1 = SmoothL1_Loss(beta=beta, reduction=reduction)

    def forward(self, 
                disparity_prediction: torch.Tensor, 
                left_img: torch.Tensor, 
                right_img: torch.Tensor, 
                reconstructured_left: torch.Tensor=None) -> torch.Tensor:
        r""" 
        Computation of the photometric loss using smooth L1 loss function in combination with the reconstructed left image.
        """
        if reconstructured_left is None:
            reconstructured_left = warp_image_by_disparity(
                right_image=right_img,
                left_disparity=disparity_prediction
            )
        return self.smooth_l1(reconstructured_left, left_img)