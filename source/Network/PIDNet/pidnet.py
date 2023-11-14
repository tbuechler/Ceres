import os
import torch
import omegaconf
import torch.nn as nn

from typing import Tuple, Type, Union
from source.Network.PIDNet.pidnet_parts import *
from source.Network.base_network import BaseNetwork

class _PIDNet(BaseNetwork):
    def __init__(
        self, 
        in_channels: int, 
        begin_channels: int,
        ppm_channels: int,
        head_channels: int,
        num_classes: int,
        blocks_num: int,
        blocks_num_deep: int
    ) -> None:
        super().__init__()
        ##################
        ## **I - Branch**
        ##################
        self.I_layer1 = nn.Sequential(*[
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=begin_channels,
                kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=begin_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=begin_channels,
                out_channels=begin_channels,
                kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=begin_channels),
            nn.ReLU(inplace=True),
        ])
        self.I_layer1_1 = self._make_layer(
            block=ResidualBlock,
            in_channels=begin_channels,
            out_channels=begin_channels,
            blocks=blocks_num
        ) ## Resolution: From 1/1 to 1/4
        self.I_layer2 = self._make_layer(
            block=ResidualBlock,
            in_channels=begin_channels,
            out_channels=begin_channels * 2,
            blocks=blocks_num, stride=2
        ) ## Resolution: From 1/4 to 1/8
        self.I_layer3 = self._make_layer(
            block=ResidualBlock,
            in_channels=begin_channels * 2,
            out_channels=begin_channels * 4,
            blocks=blocks_num_deep, stride=2
        ) ## Resolution: From 1/8 to 1/16
        self.I_layer4 = self._make_layer(
            block=ResidualBlock,
            in_channels=begin_channels * 4,
            out_channels=begin_channels * 8,
            blocks=blocks_num_deep, stride=2
        ) ## Resolution: From 1/16 to 1/32
        self.I_layer5 = self._make_layer(
            block=Bottleneck,
            in_channels=begin_channels * 8,
            out_channels=begin_channels * 8,
            blocks=2, stride=2
        ) ## Resolution: From 1/32 to 1/64

        ##################
        ## **P - Branch**
        ##################
        self.P_layer1 = self._make_layer(            
            block=ResidualBlock,
            in_channels=begin_channels * 2,
            out_channels=begin_channels * 2,
            blocks=blocks_num
        )
        self.dim_reduction1 = nn.Sequential(
            nn.Conv2d(
                in_channels=begin_channels * 4,
                out_channels=begin_channels * 2, 
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(begin_channels * 2)
        )
        self.P_Pag1 = PagFM(begin_channels * 2, begin_channels)
        self.P_layer2 = self._make_layer(            
            block=ResidualBlock,
            in_channels=begin_channels * 2,
            out_channels=begin_channels * 2,
            blocks=blocks_num
        )
        self.dim_reduction2 = nn.Sequential(
            nn.Conv2d(
                in_channels=begin_channels * 8,
                out_channels=begin_channels * 2, 
                kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(begin_channels * 2)
        )
        self.P_Pag2 = PagFM(begin_channels * 2, begin_channels)
        self.P_layer3 = self._make_layer(            
            block=Bottleneck,
            in_channels=begin_channels * 2,
            out_channels=begin_channels * 2,
            blocks=1
            )

        ##################
        ## **D - Branch**
        ##################
        if blocks_num <= 2:
            self.D_layer1 = self._make_layer(
                block=ResidualBlock,
                in_channels=begin_channels * 2,
                out_channels=begin_channels, blocks=1
            )
            self.D_layer2 = self._make_layer(
                block=Bottleneck,
                in_channels=begin_channels,
                out_channels=begin_channels, blocks=1
            )
            self.diff3 = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=begin_channels * 4, 
                    out_channels=begin_channels, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(begin_channels)
            ])
            self.diff4 = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=begin_channels * 8, 
                    out_channels=begin_channels * Bottleneck.expansion, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(begin_channels * Bottleneck.expansion)
            ])
            self.spp = PaPPM(begin_channels * 8 * Bottleneck.expansion, ppm_channels, begin_channels * 4)
            self.dfm = LightBag(begin_channels * 4, begin_channels * 4)
            self.D_layer3 = self._make_layer(            
                block=Bottleneck,
                in_channels=begin_channels * Bottleneck.expansion,
                out_channels=begin_channels * 2,
                blocks=1
                )
        else:
            self.D_layer1 = self._make_layer(
                block=ResidualBlock,
                in_channels=begin_channels * 2,
                out_channels=begin_channels * 2, blocks=1
            )
            self.D_layer2 = self._make_layer(
                block=ResidualBlock,
                in_channels=begin_channels * 2,
                out_channels=begin_channels * 2, blocks=1
            )
            self.diff3 = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=begin_channels * 4, 
                    out_channels=begin_channels * 2, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(begin_channels * 2)
            ])
            self.diff4 = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=begin_channels * 8, 
                    out_channels=begin_channels * 2, 
                    kernel_size=3, 
                    padding=1, 
                    bias=False
                ),
                nn.BatchNorm2d(begin_channels * 2)
            ])
            self.spp = DaPPM(begin_channels * 8 * Bottleneck.expansion, ppm_channels, begin_channels * 4)
            self.dfm = Bag(begin_channels * 4, begin_channels * 4)
            self.D_layer3 = self._make_layer(            
                block=Bottleneck,
                in_channels=begin_channels * 2,
                out_channels=begin_channels * 2,
                blocks=1
                )

        ##################
        ## **Heads**
        ##################
        ## During training we have three different heads while only
        ## the segmentation head stays during inference.
        if self.training:
            self.p_head = S_B_Head(
                in_channels=begin_channels * 2,
                mid_channels=head_channels,
                out_channels=num_classes,
                scale_factor=8
            ) 
            self.d_head = S_B_Head(
                in_channels=begin_channels * 2,
                mid_channels=head_channels,
                out_channels=1,
                scale_factor=8
            )
        self.seg_head = S_B_Head(
                in_channels=begin_channels * 4,
                mid_channels=head_channels,
                out_channels=num_classes,
                scale_factor=8
            )

    def _make_layer(self,
        block: Type[Union[ResidualBlock, Bottleneck]],
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int=1
        ):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(*[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels*block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(num_features=out_channels*block.expansion)
            ])

        layers = []
        layers.append(
            block(
                in_channels=in_channels, 
                out_channels=out_channels, 
                stride=stride,
                downsample=downsample
            )
        )
        in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    stride=1,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        r""" Forward call of the network. """
        d_interpolate_shape = (
            torch.div(x.shape[2], 8, rounding_mode='floor'),
            torch.div(x.shape[3], 8, rounding_mode='floor')
        )
        ## I Branch
        i_out1 = self.I_layer1(x)
        i_out2 = self.I_layer1_1(i_out1)
        i_out3 = self.I_layer2(i_out2) # Here: 1/4
        i_out4 = self.I_layer3(i_out3) # Here: 1/8 and #channel: begin_channels * 2
        i_out5 = self.I_layer4(i_out4) # Here: 1/16 and #channel: begin_channels * 4
        i_out6 = self.I_layer5(i_out5) # Here: 1/32 and #channel: begin_channels * 8
        i_out = F.interpolate(  self.spp(i_out6),
                                size=d_interpolate_shape,
                                mode='bilinear', align_corners=False) # Here: 1/64 and #channel: begin_channels * 8
        ## P Branch
        p_out1 = self.P_layer1(i_out3)
        p_out1 = self.P_Pag1(p_out1, self.dim_reduction1(i_out4)) # Do not overwrite out from here
        p_out = self.P_layer2(p_out1)
        p_out = self.P_Pag2(p_out, self.dim_reduction2(i_out5))
        p_out = self.P_layer3(p_out)
        ## D Branch 
        d_out1 = self.D_layer1(i_out3)
        d_out1 = self.D_layer2(d_out1 + F.interpolate(
                                        self.diff3(i_out4),
                                        size=d_interpolate_shape,
                                        mode='bilinear', align_corners=False))
        d_out1 = d_out1 + F.interpolate(self.diff4(i_out5),
                                        size=d_interpolate_shape,
                                        mode='bilinear', align_corners=False)
        d_out = self.D_layer3(d_out1)

        ## Seg Head
        seg_out = self.seg_head(self.dfm(p_out, i_out, d_out))

        if self.training:
            return [self.p_head(p_out1), seg_out, self.d_head(d_out1)]
        else:
            return seg_out

    def export(self, input_args: Tuple[torch.Tensor, ], dir_path: str):
        r"""
        Export of checkpoint and ONNX model.

        Args:

        * `input_args (Tuple[torch.Tensor, ])`: 
            * Random input argument for tracing.
        * `dir_path (str)`: 
            * Path the models are saved to.
        """
        assert not self.training, "[PIDNet] Model is suppose to be in eval mode."
        assert os.path.isdir(dir_path), "[PIDNet] dir_path is not a directory: {}".format(dir_path)

        _input_names  = ['i_img']
        _output_names = ['o_seg']
        _dynamic_axes = None

        file_path = os.path.join(dir_path, str(self.__class__.__name__ + ".onnx"))
        torch.onnx.export(
            model           = self,
            args            = input_args,
            f               = file_path,
            export_params   = True,
            verbose         = False,
            input_names     = _input_names,
            output_names    = _output_names,
            dynamic_axes    = _dynamic_axes,
            opset_version   = 11
        )


class PIDNet(_PIDNet):
    r"""
    Wrapper class for the PIDNet class that takes as an argument an omegaconf dictionary to make the network architecture adaptable from the configuration file itself.
    """
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        super().__init__(
            in_channels     =   cfg.network.attributes.in_channels,
            begin_channels  =   cfg.network.attributes.begin_channels,
            ppm_channels    =   cfg.network.attributes.ppm_channels,
            head_channels   =   cfg.network.attributes.head_channels,
            num_classes     =   cfg.network.attributes.num_classes,
            blocks_num      =   cfg.network.attributes.blocks_num,
            blocks_num_deep =   cfg.network.attributes.blocks_num_deep
        )
