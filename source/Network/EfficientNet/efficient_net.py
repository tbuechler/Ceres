import os,sys
import omegaconf
import torch
import torch.nn as nn

from typing import Tuple
from source.Network.Activation.swish import Swish
from source.Network.EfficientNet.efficient_net_parts import MBConv
from source.Network.EfficientNet.efficient_net_utils import get_model_params, round_filters, round_repeats
from source.Network.base_network import BaseNetwork


class _EfficientNet(BaseNetwork):
    """
    Generic backbone network architecture for the family of models that are described in "EfficientNet: Rethinking Model Scaling for Convolution Neural Networks".

    Args:
        compound_coefficient (int): Coefficient which uniformly scales network depth, width and resolution.
        in_channels (int): Number of channels of incoming image.
    """
    def __init__(self, compound_coefficient: int, in_channels: int) -> None:
        super().__init__()
        assert (0 <= compound_coefficient <= 7), "[EfficientNet] Compound efficient must be within 0 and 7."

        self.in_channels = in_channels

        ## Global architecture and block arguments and parameters
        blocks_arguments, global_params = get_model_params(compound_coefficient=0)

        ## Stage 1 (Stem)
        _num_channels = round_filters(filters=32, width_coefficient=global_params.width_coefficient)
        self._stem = nn.Sequential(*[
            nn.Conv2d(in_channels=self.in_channels, out_channels=_num_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=_num_channels, eps=global_params.batch_norm_epsilon, momentum=global_params.batch_norm_momentum),
            Swish()
        ])

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in blocks_arguments:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params.width_coefficient),
                output_filters=round_filters(block_args.output_filters, global_params.width_coefficient),
                num_repeat=round_repeats(block_args.num_repeat, global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(
                MBConv(
                    in_channels=block_args.input_filters,
                    out_channels=block_args.output_filters,
                    kernel_size=block_args.kernel_size,
                    stride=block_args.stride,
                    exp_ratio=block_args.expand_ratio,
                    se_ratio=block_args.se_ratio
                )
            )
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):           
                self._blocks.append(
                    MBConv(
                        in_channels=block_args.input_filters,
                        out_channels=block_args.output_filters,
                        kernel_size=block_args.kernel_size,
                        stride=block_args.stride,
                        exp_ratio=block_args.expand_ratio,
                        se_ratio=block_args.se_ratio
                    )
                )

    def return_feature_output_channels(self):
        channels = []
        for idx, block in enumerate(self._blocks):
            if block.dw_conv.stride[0] == 2:
                channels.append(block.in_channels)
            if idx == len(self._blocks) - 1:
                channels.append(block.out_channels)
        return channels

    def return_feature_pyramid_level(self):
        return [1, 2, 3, 4, 5]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass for EfficientNet Backbone.

        Parameters
        ----------
        x: torch.Tensor
            Torch tensor with shape (BxCxWxH)

        Returns
        ----------
        List of four feature vectors at different scales.
            [1/4, 1/8, 1/16, 1/32]
        """
        ## Stem
        x = self._stem(x)
        
        ## Blocks
        feature_maps = []
        for idx, block in enumerate(self._blocks):
            if block.dw_conv.stride[0] == 2:
                feature_maps.append(x)
            x = block(x)
            
            if idx == len(self._blocks) - 1:
                feature_maps.append(x)

        return feature_maps
    
    def export(self, input_args: Tuple[torch.Tensor,], dir_path: str):
        ## Precheck
        assert os.path.isdir(dir_path), "[EfficientNet] dir_path is not a directory: {}".format(dir_path)

        ## ONNX export
        # Define attributes for ONNX export
        _input_names  = ['i_x']
        _output_names = ['o_p1', 'o_p2', 'o_p3', 'o_p4', 'o_p5']
        _dynamic_axes = None

        # Precheck parameter
        assert len(input_args) == len(_input_names), \
            "[ONNX - EfficientNet] Length of input arguments must match length of input names."

        assert input_args[0].size()[1] == self.in_channels, \
            "[ONNX - EfficientNet] Channel size of input arg ({}) does not match with architecture ({}).".format(
                input_args[0].size()[1], self.in_channels
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

    def init_weights(self):
        raise NotImplementedError

class EfficientNet(_EfficientNet):
    r""" Wrapper class for EfficientNet to call it by using config file. """
    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        self.cfg = cfg
        super().__init__(
            self.cfg.network.compound_coefficient, 
            self.cfg.network.in_channels
            )