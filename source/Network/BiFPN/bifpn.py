import os
import torch
import torch.nn as nn

from typing import Tuple
from source.Network.Activation.swish import Swish
from source.Network.base_network import BaseNetwork
from source.Network.BiFPN.bifpn_parts import DepthwiseSeparableConvolution


## BiFPN design according to the paper: "EfficientDet: Scalable and Efficient Object Detection"
## See page 3, figure 2(d)
# 
#        ┌───────────┬──────────────┐
#        │ Bottom-Up │   Top-Down   │
#        ├───────────┴──────────────┤
#        │                          │
#   ┌──┐ │               ┌────┐     │
#   │P7├─┼─────┬────────►│P7_2├─────┼─►
#   └──┘ │     │         └─▲──┘     │
#        │     │           │        │
#        │     │           └────┐   │
#        │ ┌───┼──────────┐     │   │
#        │ │   │          │     │   │
#   ┌──┐ │ │  ┌▼───┐     ┌▼───┐ │   │
#   │P6├─┼─┴─►│P6_1├────►│P6_2├─┴───┼─►
#   └──┘ │    └┬───┘     └─▲──┘     │
#        │     │           │        │
#        │     │           └────┐   │
#        │ ┌───┼──────────┐     │   │
#        │ │   │          │     │   │
#   ┌──┐ │ │  ┌▼───┐     ┌▼───┐ │   │
#   │P5├─┼─┴─►│P5_1├────►│P5_2├─┴───┼─►
#   └──┘ │    └┬───┘     └─▲──┘     │
#        │     │           │        │
#        │     │           └────┐   │
#        │ ┌───┼──────────┐     │   │
#        │ │   │          │     │   │
#   ┌──┐ │ │  ┌▼───┐     ┌▼───┐ │   │
#   │P4├─┼─┴─►│P4_1├────►│P4_2├─┴───┼─►
#   └──┘ │    └┬───┘     └─▲──┘     │
#        │     │           │        │
#        │     │           └────┐   │
#        │     └──────────┐     │   │
#        │                │     │   │
#   ┌──┐ │               ┌▼───┐ │   │
#   │P3├─┼──────────────►│P3_2├─┴───┼─►
#   └──┘ │               └────┘     │
#        │                          │
#        └──────────────────────────┘
class BiFPN_Layer(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()

        self.epsilon = 1e-3 # TODO: Replace Magic Number by argument
        self.num_features = num_features

        ## Convolution layer
        # Bottom-Up path
        self.P6_1_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P5_1_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P4_1_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        # Top-Down path
        self.P3_2_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P4_2_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P5_2_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P6_2_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        self.P7_2_conv = DepthwiseSeparableConvolution(in_channels=self.num_features, out_channels=self.num_features)
        # Activation function for inputs (?)
        self.activation_convs = Swish()

        ## Resize functionality
        # Upsample
        self.upsample_p7_to_p6 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p6_to_p5 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p5_to_p4 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_p4_to_p3 = nn.Upsample(scale_factor=2, mode='nearest')
        # Downsample
        self.downsample_p3_to_p4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.downsample_p4_to_p5 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.downsample_p5_to_p6 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.downsample_p6_to_p7 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        ## Weights
        ## See page 3 in paper under Weighted Feature Fusion
        ## "... we propose to add an additional weight for each input ..."
        ## Apply ReLU function to each of them in _forward(...)
        # Bottom-Up path
        self.P6_1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) # Two inputs and requires_grad = true (learnable)
        self.P5_1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) # Two inputs and requires_grad = true (learnable)
        self.P4_1_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) # Two inputs and requires_grad = true (learnable)
        # Top-Down path
        self.P3_2_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) # Two inputs and requires_grad = true (learnable)
        self.P4_2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) # Three inputs and requires_grad = true (learnable)
        self.P5_2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) # Three inputs and requires_grad = true (learnable)
        self.P6_2_weight = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True) # Three inputs and requires_grad = true (learnable)
        self.P7_2_weight = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True) # Two inputs and requires_grad = true (learnable)
        # Activation function for weights
        self.activation_weights = nn.ReLU()

    def forward(self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor,
        p6: torch.Tensor,
        p7: torch.Tensor) -> torch.Tensor:
        
        # P6_1
        P6_1_weight = self.activation_weights(self.P6_1_weight)
        weights = P6_1_weight / (torch.sum(P6_1_weight, dim=0) + self.epsilon)
        P6_1 = self.P6_1_conv(
            self.activation_convs(weights[0] * p6 + weights[1] * self.upsample_p7_to_p6(p7))
        )
        # P5_1
        P5_1_weight = self.activation_weights(self.P5_1_weight)
        weights = P5_1_weight / (torch.sum(P5_1_weight, dim=0) + self.epsilon)
        P5_1 = self.P5_1_conv(
            self.activation_convs(weights[0] * p5 + weights[1] * self.upsample_p6_to_p5(P6_1))
        )
        # P4_1
        P4_1_weight = self.activation_weights(self.P4_1_weight)
        weights = P4_1_weight / (torch.sum(P4_1_weight, dim=0) + self.epsilon)
        P4_1 = self.P4_1_conv(
            self.activation_convs(weights[0] * p4 + weights[1] * self.upsample_p5_to_p4(P5_1))
        )
        # P3_2
        P3_2_weight = self.activation_weights(self.P3_2_weight)
        weights = P3_2_weight / (torch.sum(P3_2_weight, dim=0) + self.epsilon)
        P3_2 = self.P3_2_conv(
            self.activation_convs(weights[0] * p3 + weights[1] * self.upsample_p4_to_p3(P4_1))
        )

        # P4_2
        P4_2_weight = self.activation_weights(self.P4_2_weight)
        weights = P4_2_weight / (torch.sum(P4_2_weight, dim=0) + self.epsilon)
        P4_2 = self.P4_2_conv(
            self.activation_convs(weights[0] * p4 + weights[1] * P4_1 + weights[2] * self.downsample_p3_to_p4(P3_2))
        )
        # P5_2
        P5_2_weight = self.activation_weights(self.P5_2_weight)
        weights = P5_2_weight / (torch.sum(P5_2_weight, dim=0) + self.epsilon)
        P5_2 = self.P5_2_conv(
            self.activation_convs(weights[0] * p5 + weights[1] * P5_1 + weights[2] * self.downsample_p4_to_p5(P4_2))
        )
        # P6_2
        P6_2_weight = self.activation_weights(self.P6_2_weight)
        weights = P6_2_weight / (torch.sum(P6_2_weight, dim=0) + self.epsilon)
        P6_2 = self.P6_2_conv(
            self.activation_convs(weights[0] * p6 + weights[1] * P6_1 + weights[2] * self.downsample_p5_to_p6(P5_2))
        )
        # P7_2
        P7_2_weight = self.activation_weights(self.P7_2_weight)
        weights = P7_2_weight / (torch.sum(P7_2_weight, dim=0) + self.epsilon)
        P7_2 = self.P7_2_conv(
            self.activation_convs(weights[0] * p7 + weights[1] * self.downsample_p6_to_p7(P6_2))
        )

        return P3_2, P4_2, P5_2, P6_2, P7_2


class BiFPN_EntryLevel(nn.Module):
    """
    Necessary module to scale down common backbone networks which
    usually provide feature maps up to a scale of 1/32 only.

    Additionally, feature dimension is scaled down/up to certain feature value.

    Parameters
    ---------
    in_channels: list[int] 
        List of channel values for each input.
    
    bifpn_channel: int
        Required channel value for the entry level of BiFPN.
    """
    def __init__(self, in_channels: int, bifpn_channel: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        assert len(in_channels) == 3, \
            "[BiFPN_EntryLevel] Expected length of in_channels to be three.\nFeature Maps from resolution 1/8, 1/16 and 1/32 shall be given."

        ## P3-P5
        self.feat_reduction = nn.ModuleList([
            nn.Sequential(*[
                nn.Conv2d(
                    in_channels=in_channel, 
                    out_channels=bifpn_channel, 
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=bifpn_channel)
            ])
            for in_channel in self.in_channels
        ])

        ## P6
        self.create_p6 = nn.Conv2d(
            in_channels=self.in_channels[-1], 
            out_channels=bifpn_channel, 
            kernel_size=3,
            stride=2,
            padding=1
        )

        ## P7
        self.create_p7 = nn.Sequential(*[
            nn.ReLU(),
            nn.Conv2d(
                in_channels=bifpn_channel, 
                out_channels=bifpn_channel, 
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        p6 = self.create_p6(p5)
        p7 = self.create_p7(p6)

        p3 = self.feat_reduction[0](p3)
        p4 = self.feat_reduction[1](p4)
        p5 = self.feat_reduction[2](p5)
        return p3, p4, p5, p6, p7


class BiFPN(BaseNetwork):
    def __init__(self, in_channels: int, num_channels: int, num_levels: int) -> None:
        super().__init__()
        self.in_channels    = in_channels
        self.num_channels   = num_channels
        self.num_levels     = num_levels

        ## Entry level of BiFPN architecture
        self.entry = BiFPN_EntryLevel(in_channels=self.in_channels, bifpn_channel=self.num_channels)

        ## Additional levels of BiFPN structure
        self.bifpn_level = nn.ModuleList([BiFPN_Layer(num_features=num_channels) for _ in range(self.num_levels)])

    def forward(self, p3: torch.Tensor, p4: torch.Tensor, p5: torch.Tensor) -> torch.Tensor:
        p3, p4, p5, p6, p7 = self.entry(p3, p4, p5)
        for level in self.bifpn_level:
            p3, p4, p5, p6, p7 = level(p3, p4, p5, p6, p7)
        return p3, p4, p5, p6, p7
        
    def export(self, input_args: Tuple[torch.Tensor], dir_path: str):
        """
        Export of checkpoint and ONNX model. 

        Args:
            input_args (Tuple[torch.Tensor]): Random input argument for tracing.
            dir_path (str): Path the models are saved to.
        """
        ## Precheck
        assert os.path.isdir(dir_path), "[BiFPN] dir_path is not a directory: {}".format(dir_path)

        # Define attributes for ONNX export
        _input_names  = ['i_p3', 'i_p4', 'i_p5']
        _output_names = ['o_p3', 'o_p4', 'o_p5', 'o_p6', 'o_p7']
        _dynamic_axes = None

        # Precheck parameter
        assert len(input_args) == len(_input_names), \
            "[ONNX - BiFPN] Length of input arguments must match length of input names."

        for _in, _ch in zip(input_args, self.in_channels):
            assert _in.size()[1] == _ch, \
                "[ONNX - BiFPN] Channel size of input arg ({}) does not match with architecture ({}).".format(
                    _in.size()[1], _ch
                )

        file_path = os.path.join(dir_path, str(self.__class__.__name__ + ".onnx"))
        torch.onnx.export(
            model           = self,
            args            = input_args,   # input
            f               = file_path,
            export_params   = True,          # Set to true, otherwise untrained model is exported.
            verbose         = False,
            input_names     = _input_names,
            output_names    = _output_names,
            dynamic_axes    = _dynamic_axes,
            opset_version   = 11             # Overwrite default 9, because of interesting new features
        )

        ## Checkpoint export 
        torch.save(self.state_dict(), file_path.replace(".onnx", ".pth"))
