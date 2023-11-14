import torch
import torch.nn as nn
import torch.nn.functional as F
from source.Network.Activation.swish import Swish


class MBConv(nn.Module):
    """
    Description
    ----------
    Mobile inverted bottleneck convolution block.
    Definition and description can be found here
        1. "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
        2. "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

    Parameters
    ----------
    in_channels:    int
        Input number of channels
    out_channels:   int
        Output number of channels
    kernel_size:    int
        Kernel size of depthwise convolution
    stride:         int
        Stride value of depthwise convolution
    exp_ratio:      float
        Expansion ratio
    se_ratio:       float
        Squeeze ratio for squeeze-and-excitation part
    """
    def __init__(self, 
        in_channels:    int,
        out_channels:   int,
        kernel_size:    int,
        stride:         int,
        exp_ratio:      float=None,
        se_ratio:       float=None
    ) -> None:
        super().__init__()
        self.exp_ratio      = exp_ratio
        self.se_ratio       = se_ratio
        self.in_channels    = in_channels
        self.out_channels   = out_channels

        ## 0. Activation function
        self.activation = Swish()

        ## 1. Expansion phase
        _channels = self.in_channels
        if self.exp_ratio != 1.:
            _channels *= self.exp_ratio
            self.exp_conv = nn.Conv2d(in_channels=self.in_channels, out_channels=_channels, kernel_size=1, bias=False)
            self.exp_bn   = nn.BatchNorm2d(num_features=_channels)
            
        ## 2. Depthwise convolution phase
        self.dw_conv = nn.Conv2d(in_channels=_channels, out_channels=_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=_channels, bias=False)
        self.dw_bn   = nn.BatchNorm2d(num_features=_channels)

        ## 3. Squeeze & Excitation phase
        if self.se_ratio:
            _squeezed_channels = max(1, int(self.in_channels * se_ratio)) # Avoid zero and negative number
            self.se_active   = nn.Conv2d(in_channels=_channels, out_channels=_squeezed_channels, kernel_size=1)
            self.se_deactive = nn.Conv2d(in_channels=_squeezed_channels, out_channels=_channels, kernel_size=1)

        ## 4. Final output phase
        self.out_conv = nn.Conv2d(in_channels=_channels, out_channels=self.out_channels, kernel_size=1, bias=False)
        self.out_bn   = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ## Save for skip connection
        _x = x

        ## Expansion and depthwise conv.
        if self.exp_ratio != 1.:
            _x = self.exp_conv(_x)
            _x = self.exp_bn(_x)
            _x = self.activation(_x)

        ## Depthwise convolution
        _x = self.dw_conv(_x)
        _x = self.dw_bn(_x)
        _x = self.activation(_x)

        ## Squeeze-and-Excitation
        if self.se_ratio:
            _x_sq = F.adaptive_avg_pool2d(_x, 1)
            _x_sq = self.se_active(_x_sq)
            _x_sq = self.activation(_x_sq)
            _x_sq = self.se_deactive(_x_sq)
            _x = torch.sigmoid(_x_sq) * _x

        ## Output
        _x = self.out_conv(_x)
        _x = self.out_bn(_x)

        ## Skip connection
        if self.in_channels == self.out_channels:
            _x += x

        return _x
