import os
import torch
import omegaconf

from typing import Tuple
from source.Logger.logger import log_warning
from source.Network.BiFPN.bifpn import BiFPN
from source.Network.base_network import BaseNetwork
from source.Utility.ObjectDetection2D.anchors import Anchors
from source.Network.EfficientNet.efficient_net import _EfficientNet
from source.Network.EfficientDet.efficient_det_parts import OD_Classifier, OD_Regressor


class _EfficientDet(BaseNetwork):
    """
    Neural network architecture described in "EfficientDet: Scalable and Efficient Object Detection".
    Basically using EfficientNet for backbone architecture and presented BiFPN as Feature Pyramid Network.

    Args:
        in_channels (int): Number of channels of input image.
        num_classes (int): Number of classes to classify.
        compound_coefficient (int): Compound efficient for backbone EfficientNet.
        num_anchors (int): Number of anchor boxes per cell.
    """
    def __init__(self, in_channels: int, num_classes: int, compound_coefficient: int, num_anchors: int) -> None:
        super().__init__()
        assert compound_coefficient <= 7 and compound_coefficient >= 0, "[EfficientDet] Compound coefficient must be between zero and seven."
        self.num_classes = num_classes
        self.fpn_num_filters = [64, 88, 112, 160, 224, 288, 384, 384, 384]
        self.fpn_cell_repeats = [3, 4, 5, 6, 7, 7, 8, 8, 8]
        self.box_class_repeats = [3, 3, 3, 4, 4, 4, 5, 5, 5]

        ## What parts do we need?
        self.backbone   = _EfficientNet(
            compound_coefficient = compound_coefficient,
            in_channels          = in_channels
        )
        # return_feature_pyramid_level
        self.fpn        = BiFPN(
            in_channels  = self.backbone.return_feature_output_channels()[-3:],
            num_channels = self.fpn_num_filters[compound_coefficient],
            num_levels   = self.fpn_cell_repeats[compound_coefficient]
        )
        
        ## OD related stuff
        self.num_anchor_boxes   = num_anchors
        self.regressor          = OD_Regressor(
            in_channels=self.fpn_num_filters[compound_coefficient],
            num_anchors=self.num_anchor_boxes,
            num_layers=self.box_class_repeats[compound_coefficient]
        )
        self.classifier   = OD_Classifier(
            in_channels = self.fpn_num_filters[compound_coefficient],
            num_anchors = self.num_anchor_boxes,
            num_layers  = self.box_class_repeats[compound_coefficient],
            num_classes = self.num_classes
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        r"""
        Forward path for EfficientDet.

        Args:
            x (torch.Tensor): Image tensor of shape [B, C, H, W].

        Returns:
            features (list[torch.Tensor]): List of different feature maps from backbone network.
            regression (list[torch.Tensor]): List of regression results of all anchor boxes of shape [B, num_anchors, 4].
            classification (list[torch.Tensor]): List of classification result of all anchor boxes of shape [B, num_anchors, 4].
        """
        ## Create feature maps
        features = self.backbone(x)
        features = self.fpn(p3 = features[-3], p4 = features[-2], p5 = features[-1])

        ## Regression and Classification
        regression      = torch.cat([self.regressor(f) for f in features],  dim=1)
        classification  = torch.cat([self.classifier(f) for f in features], dim=1)

        return features, regression, classification

    def export(self, input_args: Tuple[torch.Tensor, ], dir_path: str):
        r"""
        Export of checkpoint and ONNX model. Models of sub-networks like Backbone and FPN are also exported.

        Args:
            input_args (Tuple[torch.Tensor, ]): Random input argument for tracing.
            dir_path (str): Path the models are saved to.
        """
        ## Precheck
        assert os.path.isdir(dir_path), "[EfficientDet] dir_path is not a directory: {}".format(dir_path)

        ## Define attributes for ONNX export
        _input_names  = ['i_img']
        _output_names = ['o_feat', 'o_regr', 'o_clas']
        _dynamic_axes = None

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

        ## Try to export submodules as well
        submodule_dir = os.path.join(dir_path, "subs")
        # Backbone
        self.backbone.export(input_args=input_args, dir_path=submodule_dir)
        # FPN
        features = self.backbone(*input_args)
        self.fpn.export(input_args=(features[-3], features[-2], features[-1]), dir_path=submodule_dir)

        ## Dump anchor meta file as well
        yaml_file_path = os.path.join(dir_path, "meta_anchor_boxes.yaml")
        self.anchor_obj.export_meta_file(file_path=yaml_file_path)

    @staticmethod
    def decode_output(anchor_boxes: torch.Tensor, regression: torch.Tensor) -> torch.Tensor:
        """
        Decodes the real bounding boxes (image plane) from the actual network output. The network output relies on the actual used loss function.
        Final format: (x1, y1, x2, y2).

        Args:
            anchor_boxes (torch.Tensor): Precomputed set of anchor boxes of shape (x1, y1, x2, y2).
            regression (torch.Tensor): Net output for the regression part. Bounding box locations with shape of (dx, dy, dHeight, dWidth).
        
        Returns:
            decoded_bb (torch.Tensor): Decoded bounding boxes based on the anchors boxes and the relative box structure from the net output.
        """
        ## Transform the anchor boxes from (x1, y1, x2, y2) to (xc, yc, height, width).
        anchor_yc       = (anchor_boxes[..., 1] + anchor_boxes[..., 3]) / 2
        anchor_xc       = (anchor_boxes[..., 0] + anchor_boxes[..., 2]) / 2
        anchor_height   = (anchor_boxes[..., 3] - anchor_boxes[..., 1])
        anchor_width    = (anchor_boxes[..., 2] - anchor_boxes[..., 0])

        ## Compute (xc, yc, height, width) for the predicted bounding box structure using anchor box as relative structure.
        predict_xc = regression[..., 0] * anchor_width + anchor_xc
        predict_yc = regression[..., 1] * anchor_height + anchor_yc

        predict_height  = regression[..., 2].exp() * anchor_height
        predict_width   = regression[..., 3].exp() * anchor_width


        ymin = predict_yc - predict_height / 2.
        xmin = predict_xc - predict_width  / 2.
        ymax = predict_yc + predict_height / 2.
        xmax = predict_xc + predict_width  / 2.

        return torch.stack([xmin, ymin, xmax, ymax], dim=2)


class EfficientDet(_EfficientDet):
    r"""
    Wrapper class for EfficientDet to call it by using config file.
    
    Args:
        cfg (omegaconf.DictConfig): Configuration dictionary created by Hydra based on the given configuration .yaml file. 
    """
    def __init__(self, cfg):
        ## Anchor boxes precomputation
        # Ratios
        try:
            _anchor_ratios = cfg.network.anchor_ratios
            assert isinstance(_anchor_ratios, list), "[EfficientDet] Anchor ratios must be a list with content of [(h1, w1), (h2, w2), ...]."
            assert ([isinstance(x, Tuple) for x in _anchor_ratios]).all(), "[EfficientDet] Anchor ratios must be a list with content of [(h1, w1), (h2, w2), ...]."
        except omegaconf.errors.ConfigAttributeError:
            _anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
            log_warning("[EfficientDet] No anchor ratios given. Using default set from COCO dataset instead: {}.".format(_anchor_ratios))
        # Scales
        try:
            _anchor_scales = cfg.network.anchor_scales
            assert isinstance(_anchor_ratios, list), "[EfficientDet] Anchor scales must be a list with content of [s1, s2, ...]."
        except omegaconf.errors.ConfigAttributeError:
            _anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            log_warning("[EfficientDet] No anchor scales given. Using default set from COCO dataset instead: {}.".format(_anchor_scales))
        # Default scale
        try:
            _anchor_scale = cfg.network.anchor_scale
            assert isinstance(_anchor_ratios, float), "[EfficientDet] Anchor scale must be a float."
        except omegaconf.errors.ConfigAttributeError:
            _anchor_scale = 4.0
            log_warning("[EfficientDet] No anchor scale given. Using default value from COCO dataset instead: {}.".format(_anchor_scale))

        ## Compute number of anchor boxes per cell by length of ratio and scales
        _num_anchors = len(_anchor_scales) * len(_anchor_ratios)

        super().__init__(
            compound_coefficient=cfg.network.attributes.compound_coefficient, 
            in_channels     =   cfg.network.attributes.in_channels, 
            num_classes     =   cfg.network.attributes.num_classes, 
            num_anchors     =   _num_anchors
        )

        self.anchor_obj = Anchors(
            anchor_scale    =   _anchor_scale,
            anchor_ratios   =   _anchor_ratios,
            anchor_scales   =   _anchor_scales,
            pyramid_levels  =   [l + 2 for l in self.backbone.return_feature_pyramid_level()], # Use [3, 4, 5, 6, 7]
            image_size      =   (cfg.network.attributes.image_height, cfg.network.attributes.image_width)
        )
        self.anchors           = self.anchor_obj()
