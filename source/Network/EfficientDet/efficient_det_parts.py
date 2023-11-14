import torch
import torch.nn as nn
from source.Network.Activation.swish import Swish


class OD_Regressor(nn.Module):
    r"""
    Regression module where for each anchor box in each cell the bounding box parameters are computed.

    Args:
        in_channels (int): Channel dimension of feature map.
        num_anchors (int): Number of anchor boxes per cell.
        num_layer (int): Number of Conv iteration/layer within regression.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_layers: int) -> None:
        super().__init__()
        self.in_channels    = in_channels
        self.num_anchors    = num_anchors
        self.num_layers     = num_layers
        # See in loss function:
        #   targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
        #   targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
        #   targets_dw = torch.log(gt_widths / anchor_widths_pi)
        #   targets_dh = torch.log(gt_heights / anchor_heights_pi)
        self.bb_attributes  = 4         

        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Conv2d(
                in_channels  = in_channels,
                out_channels = in_channels,
                kernel_size  = 3,
                padding      = 1,
                stride       = 1
            )),
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

        self.head = nn.Conv2d(
            in_channels  = self.in_channels,
            out_channels = self.num_anchors * self.bb_attributes,
            kernel_size  = 3,
            padding      = 1,
            stride       = 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C, H, W)
        # (B, C, H, W) -> (B, num_achors * bb_attributes, H, W)
        x = self.layers(x)
        x = self.head(x)

        ## (B, C, H, W) -> (B, H, W, num_anchors * bb_attributes)
        x = x.permute(0, 2, 3, 1)
        ## (B, H, W, num_anchors * bb_attributes) ->  (B, (height*width*num_anchors),bb_attributes)   
        ## Per cell in grid (height*width)*num_achors we have box_attributes
        x = x.contiguous().view(x.shape[0], -1, 4)
        return x

class OD_Classifier(nn.Module):
    r"""
    Classification path.

    Args:
        in_channels (int): Channel dimension of feature map.
        num_anchors (int): Number of anchors per cell.
        num_layer (int): Number of Conv iteration/layer within classification.
        num_classes (int): Number of different classes for each cell and anchor.
    """
    def __init__(self, in_channels: int, num_anchors: int, num_layers: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels    = in_channels
        self.num_anchors    = num_anchors
        self.num_layers     = num_layers
        self.num_classes    = num_classes

        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Conv2d(
                in_channels  = in_channels,
                out_channels = in_channels,
                kernel_size  = 3,
                padding      = 1,
                stride       = 1,
                bias         = False
            )),
            layers.append(nn.BatchNorm2d(num_features=in_channels)),
            layers.append(Swish())
        self.layers = nn.Sequential(*layers)

        self.head = nn.Conv2d(
            in_channels  = self.in_channels,
            out_channels = self.num_anchors * self.num_classes,
            kernel_size  = 3,
            padding      = 1,
            stride       = 1
        )

    def forward(self, x: torch.Tensor):
        ## (B, C, H, W) -> (B, C, H, W)
        ## (B, C, H, W) -> (B, num_achors * num_classes, H, W)
        x = self.layers(x)
        x = self.head(x)

        ## (B, C, H, W) -> (B, H, W, num_anchors * num_classes)
        x = x.permute(0, 2, 3, 1)
        ## (B, C, H, W) -> (B, H, W, num_anchors, num_classes)
        x = x.contiguous().view(
            x.shape[0], x.shape[1], x.shape[2], self.num_anchors, self.num_classes
        )
        ## (B, C, H, W) -> (B, H*W*num_anchors, num_classes)
        x = x.contiguous().view(x.shape[0], -1, self.num_classes)
        return x.sigmoid()
