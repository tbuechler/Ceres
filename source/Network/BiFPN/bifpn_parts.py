import torch
import torch.nn as nn
from source.Network.Activation.swish import Swish


## In the paper it is mentioned: "... we use depthwise separable convolution for feature fusion, and add batch normalization and activation"
## Depthwise separable convolution operation -> a depthwise convolution followed by a pointwise convolution
##
## From different paper "Xception: Deep Learning with Depthwise Separable Convolutions":
##  "A depthwise convolution separable convolution, commonly called separable convolution in deep learning frameworks,
##   consists in a depthwise convolution, i.e. a spatial convolution performed independently over each channel of an input,
##   followed by a pointwise convolution, i.e. a 1x1 convolution, projecting the channels output by the depthwise convolution
##   onto a new channel space."
class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        ## Separable Convolution
        # depthwise
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            groups=in_channels,
            padding=1,
            bias=False
        )
        # pointwise
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
            bias=False
        )

        # bn
        self.batchnorm = nn.BatchNorm2d(
            num_features=out_channels,
            momentum=0.01,
            eps=1e-3
        )

        # activation: In the paper it is mentioned that Swish is used.
        self.activation = Swish()

    def forward(self, x: torch.Tensor):
        # do conv
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        # do bn
        x = self.batchnorm(x)
        # do activation
        return self.activation(x)

