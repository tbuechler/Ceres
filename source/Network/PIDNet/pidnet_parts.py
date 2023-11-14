import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class Bottleneck(nn.Module):
    expansion: int = 2
    def __init__(self, in_channels: int, out_channels, stride: int, downsample: Optional[nn.Module]=False) -> None:
        super().__init__()

        self.seq1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ])

        self.seq2 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False, 
                stride=stride
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ])

        self.seq3 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * self.expansion,
                kernel_size=1,
                bias=False, 
            ),
            nn.BatchNorm2d(num_features=out_channels * self.expansion)
        ])

        self.downsample         = downsample
        self.last_activation    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.seq1(x)
        out = self.seq2(out)
        out = self.seq3(out)

        if self.downsample:
            out += self.downsample(x)
        else:
            out += x

        return self.last_activation(out)


class ResidualBlock(nn.Module):
    r"""
    The entire network are developed [...] which adopted cascaded residual blocks as backbone, for hardware-friendly architecture.
    """
    expansion: int = 1

    def __init__(self, in_channels: int, out_channels, stride: int, downsample: Optional[nn.Module]=None) -> None:
        super().__init__()

        self.seq1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        ])

        self.seq2 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ])

        self.downsample         = downsample
        self.last_activation    = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        out = self.seq1(x)
        out = self.seq2(out)

        if self.downsample:
            out += self.downsample(x)
        else:
            out += x

        return self.last_activation(out)


class S_B_Head(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.conv1 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ])

        self.conv2 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=True
            )
        ])
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.interpolate(
            x,
            size=(x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor),
            mode='bilinear',
            align_corners=False
        )
        return x


class LightBag(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_comp_P = nn.Sequential(*[
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ])

        self.conv_comp_I = nn.Sequential(*[
            nn.Conv2d(
                kernel_size=1,
                in_channels=in_channels,
                out_channels=out_channels,
                bias=False
            ),
            nn.BatchNorm2d(num_features=out_channels)
        ])

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor):
        sigma = torch.sigmoid(d)
        dp = p * sigma
        di = i * (1 - sigma)
        
        p = self.conv_comp_P(p + di)
        i = self.conv_comp_I(i + dp)
        return (p + i)


class Bag(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_comp = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                kernel_size=3,
                in_channels=in_channels,
                out_channels=out_channels,
                padding=1, bias=False
            )
        ])

    def forward(self, p: torch.Tensor, i: torch.Tensor, d: torch.Tensor):
        sigma = torch.sigmoid(d)
        p = p * sigma
        i = i * (1 - sigma)
        return self.conv_comp(p + i)


class PagFM(nn.Module):
    r""" PAG Fusion module. """
    def __init__(self, in_out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.p_conv_bn = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_out_channels,
                out_channels=mid_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=mid_channels)
        ])
        self.i_conv_bn = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_out_channels,
                out_channels=mid_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(num_features=mid_channels)
        ])

    def forward(self, p: torch.Tensor, i: torch.Tensor):
        ## Path to compute sigma
        p_sigma = self.p_conv_bn(p)
        
        i_sigma = self.i_conv_bn(i)
        i_sigma = F.interpolate(i_sigma, size=p.shape[-2:],
                            mode='bilinear', align_corners=False)
        i_sigma = torch.sum((p_sigma * i_sigma), dim=1, keepdim=True)
        sigma = torch.sigmoid(i_sigma)

        ## Path to output
        i = F.interpolate(i, size=p.shape[-2:],
                            mode='bilinear', align_corners=False)

        return ((1-sigma)*p) + (sigma * i)


class DaPPM(nn.Module):
    """ Used for PIDNet-L 
    From: Deep dual-resolution networks for real-time and accurate semantic segmentation of road scenes
    
    ???
    """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                bias=False
            )
        ])

        self.branch1 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch2 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch3 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch4 = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        ## Do four conv3x3 at once by grouping
        self.proc1 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ])

        self.proc2 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ])

        self.proc3 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ])

        self.proc4 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
        ])

        self.shortcut = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        ])

        self.compression = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels * 5), ## 5 inputs
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels * 5,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        ])

    def forward(self, i: torch.Tensor):
        ## Compute first branch first, since the other depend on this
        branch0_out = self.branch0(i)

        ## Each branch step by step
        branch1_out = F.interpolate(self.branch1(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch2_out = F.interpolate(self.branch2(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch3_out = F.interpolate(self.branch3(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch4_out = F.interpolate(self.branch4(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out

        ## Concatenation of all branches
        out = self.compression(
            torch.cat([
                branch0_out,
                self.proc1(branch1_out),
                self.proc2(branch2_out),
                self.proc3(branch3_out),
                self.proc4(branch4_out)
            
            ], dim=1)
        )
        return out + self.shortcut(i)


class PaPPM(nn.Module):
    """ Used for PIDNet-S/M """
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                bias=False
            )
        ])

        self.branch1 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch2 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch3 = nn.Sequential(*[
            nn.AvgPool2d(kernel_size=17, stride=8, padding=8),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        self.branch4 = nn.Sequential(*[
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels, 
                kernel_size=1, 
                bias=False
            )
        ])

        ## Do four conv3x3 at once by grouping
        self.grouped_conv = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels*4), # 4 inputs
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels*4,
                out_channels=mid_channels*4,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=4
            )
        ])

        self.shortcut = nn.Sequential(*[
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        ])

        self.compression = nn.Sequential(*[
            nn.BatchNorm2d(num_features=mid_channels * 5), ## 5 inputs
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels * 5,
                out_channels=out_channels,
                kernel_size=1,
                bias=False
            )
        ])

    def forward(self, i: torch.Tensor):
        ## Compute first branch first, since the other depend on this
        branch0_out = self.branch0(i)

        ## Each branch step by step
        branch1_out = F.interpolate(self.branch1(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch2_out = F.interpolate(self.branch2(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch3_out = F.interpolate(self.branch3(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out
        branch4_out = F.interpolate(self.branch4(i), size=(i.shape[-2:]), mode='bilinear', align_corners=False) + branch0_out

        ## Concatenation of all branches
        out = self.grouped_conv(
            torch.cat([ 
                branch1_out,
                branch2_out,
                branch3_out,
                branch4_out
            ], dim=1)
        )

        out = self.compression(torch.cat([branch0_out, out], dim=1))
        return out + self.shortcut(i)








class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

