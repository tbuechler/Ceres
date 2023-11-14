import torch
import torch.nn as nn

from source.Loss.Common.misc_ops import gradient_x, gradient_y


class InverseDepthSmoothness(nn.Module):
    r"""
    Image-aware inverse depth smoothness loss.

    Penalizes sharp disparity/depth transition in the absence of edges in the reference image, 
    i.e. disparity map and left stereo image, according to

    $$ L_{smooth} = \frac{1}{N}\sum_{j=1}^N |\delta_x d'_j| \cdot e^{-||\delta_x I^l_j||} + 
                    \frac{1}{N}\sum_{j=1}^N |\delta_y d'_j| \cdot e^{-||\delta_y I^l_j||}$$.
    """
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, i_disparity: torch.Tensor, i_img_left: torch.Tensor):
        r""" 
        Computation of  inverse depth smoothness using left camera image and the disparity map. 
        """
        assert len(i_disparity.shape), "[InverseDepthSmoothness] Shape of incoming disparity prediction must be [B, C, H, W]."
        assert len(i_img_left.shape), "[InverseDepthSmoothness] Shape of incoming left image must be [B, C, H, W]."

        ## Compute gradients of given disparity and image tensor.
        disparity_delta_x = gradient_x(i_disparity)
        disparity_delta_y = gradient_y(i_disparity)
        left_img_delta_x  = gradient_x(i_img_left)
        left_img_delta_y  = gradient_y(i_img_left)

        ## Compute weights from image how intense disparity values will be penalized
        img_weights_x = torch.exp( 
            torch.mean(left_img_delta_x, dim=1, keepdim=True)
        ) 
        img_weights_y = torch.exp(
            torch.mean(left_img_delta_y, dim=1, keepdim=True)
        )

        assert (disparity_delta_x.shape == img_weights_x.shape), \
            "[InverseDepthSmoothness] Shape of image x weights seems to be wrong."
        assert (disparity_delta_y.shape == img_weights_y.shape), \
            "[InverseDepthSmoothness] Shape of image y weights seems to be wrong."

        ## Compute smoothness values and return the sum of the mean of both
        smoothness_x = torch.abs(disparity_delta_x * img_weights_x)
        smoothness_y = torch.abs(disparity_delta_y * img_weights_y)
        return torch.mean(smoothness_x) + torch.mean(smoothness_y)

