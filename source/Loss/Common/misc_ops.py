import torch
import torch.nn as nn


def warp_image_by_disparity(right_image: torch.Tensor, left_disparity: torch.Tensor) -> torch.Tensor:
    r"""
    Warping the right input image to the left based on the disparity output of the network by
    $$
    \text{img}_l(x, y) = \text{img}_r(x - \text{disp}_l(x, y), y).
    $$

    Args:
        right_image (torch.Tensor): Image tensor of shape [B, C, H, W].
        left_disparity (torch.Tensor): Image tensor of shape [B, 1, H, W].

    Returns:
        warped_image (torch.Tensor): Warped right image.
    """
    B, _, H, W = right_image.size()
    xx = torch.linspace(0, W - 1, W, device=right_image.device, dtype=right_image.dtype).repeat(H,1)
    yy = torch.linspace(0, H - 1, H, device=right_image.device, dtype=right_image.dtype).reshape(-1,1).repeat(1,W)

    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)

    grid = torch.cat((xx,yy),1)

    vgrid = grid

    vgrid[:,0,:,:] = 2.0*(vgrid[:,0,:,:]-left_disparity[:,0,:,:])/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0

    return nn.functional.grid_sample(right_image, vgrid.permute(0,2,3,1), align_corners=True)

def gradient_x(i_input: torch.Tensor) -> torch.Tensor:
    r""" Computes the gradient of a tensor in x direction. """
    assert len(i_input.shape) == 4, "[gradient_x] Input tensor must be of shape [B, C, H, W]."
    return i_input[:, :, :, :-1] - i_input[:, :, :, 1:]

def gradient_y(i_input: torch.Tensor) -> torch.Tensor:
    r""" Computes the gradient of tensor in y direction. """
    assert len(i_input.shape) == 4, "[gradient_y] Input tensor must be of shape [B, C, H, W]."
    return i_input[:, :, :-1, :] - i_input[:, :, 1:, :]
