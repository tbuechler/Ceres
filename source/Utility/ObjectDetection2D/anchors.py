import yaml
import torch
import itertools
import numpy as np
from typing import Tuple


class Anchors:
    """
    The purpose of this class is to generate multi-scale anchor boxes in form [x1, y1, x2, y2].
    
    Parameters
    ----------
    pyramid_levels: list[int]
        List of used pyramid levels/scales.
    image_size: Tuple[int, int]
        Resolution of input image in (Height, Width).
    anchor_scale:   float
        tbd
    anchor_scales:  list[float]
        Different scales for all anchor boxes at each stride.
    anchor_ratios:  list[Tuple[float, float]]
        Different ratios for all anchor boxes at each stride.
    """
    def __init__(self,
        pyramid_levels: list,   
        image_size:     Tuple[int, int],
        anchor_scale:   float = 4., # Not clear what this parameter is actually doing.
        anchor_scales:  list  = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)],
        anchor_ratios:  list  = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    ) -> None:
        self.pyramid_levels = pyramid_levels
        self.strides       = [2 ** x for x in self.pyramid_levels]
        self.anchor_scale  = anchor_scale
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.image_height  = image_size[0]
        self.image_width   = image_size[1]

    def __call__(self):
        boxes_all_strides = []
        for stride in self.strides:
            boxes_stride = []

            ## Iterate over all possible scales and ratios
            for scale, ratio in itertools.product(self.anchor_scales, self.anchor_ratios):
                
                ## Create cell coordinates within (scaled) image
                x = np.arange(stride / 2, self.image_width, stride)  # Start at stride/2 since zero does not make sense
                y = np.arange(stride / 2, self.image_height, stride) # Start at stride/2 since zero does not make sense
                x_mesh, y_mesh = np.meshgrid(x, y)
                x_mesh = x_mesh.reshape(-1)
                y_mesh = y_mesh.reshape(-1)

                ## Create anchor box size for each cell
                base_anchor_size = self.anchor_scale * stride * scale       # Anchor box is based on origin image, but dimension based on stride etc.
                anchor_size_y_half = base_anchor_size * ratio[1] / 2.0      # Anchor box half height
                anchor_size_x_half = base_anchor_size * ratio[0] / 2.0      # Anchor box half width

                ## Save created anchor boxes
                boxes =  np.vstack((
                    x_mesh - anchor_size_x_half,    # x1
                    y_mesh - anchor_size_y_half,    # y1
                    x_mesh + anchor_size_x_half,    # x2
                    y_mesh + anchor_size_y_half     # y2
                )) # Results in (4, (width * height)) dimension

                ## (4, (width * height)) -> ((width * height), 4)
                boxes = np.swapaxes(boxes, 0, 1)
                ## Add another dimension for vertical stacking later for different anchor versions
                ## ((width * height), 4) -> ((width * height), 1, 4)
                boxes_stride.append(np.expand_dims(boxes, axis=1))
            
            ## [((width * height), 1, 4), ...] -> ((width * height), 9, 4) where 9 is equal to different variation of scale and ratio (num_anchors per cell)
            boxes_strides = np.concatenate(boxes_stride, axis=1)
            ## List of all anchors will contain elements of shape (number_all_anchors_per_stride, 4) at ths point
            boxes_all_strides.append(boxes_strides.reshape([-1, 4]))

        ## Stack all elements vertically
        boxes_all_strides = np.vstack(boxes_all_strides)

        ## Transform all created anchor boxes to torch tensors
        anchor_boxes = torch.from_numpy(boxes_all_strides.astype(np.float32))
        anchor_boxes = anchor_boxes.unsqueeze(0)

        return anchor_boxes

    def get_num_anchors_per_cell(self) -> int:
        """
        Copmutes the number of anchor boxes per cell/pixel

        Returns
        -------
        num_anchor: int
            Number of anchor boxes for each cell/pixel.
        """
        return len(self.anchor_scales) * len(self.anchor_ratios)

    def export_meta_file(self, file_path: str):
        """ Dumps out a meta-file of the anchor boxes and the setup. """
        meta = {}
        meta['anchor_format']       = "x1,y1,x2,y2"
        meta['anchors_per_cell']    = self.get_num_anchors_per_cell()
        meta['input_height']        = self.image_height
        meta['input_width']         = self.image_width
        meta['input_strides']       = self.strides
        meta['anchor_scale']        = self.anchor_scale
        meta['anchor_scales']       = self.anchor_scales
        meta['anchor_ratios']       = {
            'height' : [x[0] for x in self.anchor_ratios],
            'width' : [x[1] for x in self.anchor_ratios]
        }
        with open(file_path, 'w') as file:
            yaml.dump(meta, file)