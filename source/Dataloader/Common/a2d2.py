import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(
        os.path.dirname(__file__), os.path.join("..", "..", "..")
    ))
    from source.Utility.visualizer import visualize_pytorch_tensor
    from source.Dataloader.Utility.dataloader import create_dataloader

import torch
import omegaconf
import numpy as np

from PIL import Image
from random import random
from source.Utility.image import *
from source.Logger.logger import log_info
from source.Utility.augmentation import AUG_TYPE
from source.Dataloader.datahandler import DataHandler
from source.Utility.visualizer import visualize_pytorch_tensor


class A2D2_ds(DataHandler):
    r"""
    # A2D2 Dataset

    Dataset for the A2D2 dataset (https://www.a2d2.audi/a2d2/en.html).
    Currently it does only support semantic segmentation or will
    only import the corresponding data, respectively. Thus it does
    not import the available LiDar information yet. 
    """
    ignore_label  = 255

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        r"""    
        Args:
        
        * `cfg (omegaconf.DictConfig)`:
            * Hydra based configuration dictionary based on the given configuration .yaml file.
            **Only the subset cfg object for the dataset is available at this point.**
        """
        super().__init__(cfg)

        ## It is possible to indicate labels to be ignored by overwriting
        ## `ignore_label`. The default value for this parameter is 255.
        try:
            self.ignore_label = self.cfg.ignore_label
        except omegaconf.errors.ConfigAttributeError:
            pass

        ## It is possible to convert the semantic segmentation labels into
        ## the cityscape format by using `use_cityscape_format`.
        try:
            self.use_cityscape_format = self.cfg.use_cityscape_format
            if self.filenames[0][3] is None:
                log_warning("Configration entry <use_cityscape_format> is set but data was not found. Unset configration entry.")
                raise omegaconf.errors.ConfigAttributeError
        except omegaconf.errors.ConfigAttributeError:
            self.use_cityscape_format = False

        if self.use_cityscape_format:
            self.class_weights = torch.FloatTensor(
                [0.8373, 0.918, 0.866, 1.0345, 
                 1.0166, 0.9969, 0.9754, 1.0489,
                 0.8786, 1.0023, 0.9539, 0.9843, 
                 1.1116, 0.9037, 1.0865, 1.0955, 
                 1.0865, 1.1529, 1.0507]
            )
        else:
            self.class_weights = torch.ones(55)

        ## Decide which label format is being used. **rgb** - it will load 
        ## the raw RGB ground truth data and convert it later into an
        ## useable format. **index** - it will load the index ground truth
        ## data. **index_cityscape** - it will load the index ground truth
        ## data which have been converted already beforehand.
        self.acceptable_label_formats = ['rgb', 'index', 'index_cityscape']

        ## If configuration entry is not given at all, by default the 
        ## RGB label images are used for ground truth generation.
        try:
            self.label_format = self.cfg.label_format
            if not (self.label_format in self.acceptable_label_formats):
                log_warning("The configuration element <label_format> ({}) is given but does not \
                             match any of the acceptable formats {}. It will be set to <rgb> and \
                             the process will be slow down probably.".format(
                    self.label_format, self.acceptable_label_formats
                ))
                raise omegaconf.errors.ConfigAttributeError
            if self.label_format == 'index' and self.use_cityscape_format:
                self.label_format = 'index_cityscape'
                log_warning("Mismatch in configuration elements. <use_cityscape_format> is used \
                             but label_format <index> is suppose to be used as well. In this  \
                             case label_format is set to <index_cityscape>.") 
            if self.label_format == 'index_cityscape' and not self.use_cityscape_format:
                if not (self.filenames[0][2] is None):
                    log_warning("Mismatch in configuration elements. <use_cityscape_format> is \
                                 not set but label_format <index_cityscape> is suppose to be  \
                                 used as well. In this case label_format is set to <index>.") 
                    self.label_format = 'index'
                else:
                    log_warning("Mismatch in configuration elements. <use_cityscape_format> is \
                                 not set but label_format <index_cityscape> is suppose to be \
                                 used as well. In this case label_format is set to <rgb>.") 
                    self.label_format = 'rgb'
        except omegaconf.errors.ConfigAttributeError:
            self.label_format = 'rgb'

    def __getitem__(self, index: int):
        r"""
        Creates a sample from the dataset for a specific pair of data 
        using an index. It will load the images using PIL, convert ground
        truth data if necessary, apply data augmentation and convert 
        sample entries into PyTorch Tensors.
                
        Returns:
            Dictionary sample in form of { 'img' : img, 'label' : label }
        """
        img = self._load_image(img_path=self.filenames[index][0])
        if self.label_format == 'rgb':
            label = self._load_label_rgb(label_path=self.filenames[index][1])
        else:
            if self.label_format == 'index':
                label = self._load_index_label(
                    label_path=self.filenames[index][2]
                )
            else:
                label = self._load_index_label(
                    label_path=self.filenames[index][3]
                )

        sample =  { 
            'img' : img, 
            'label' : label 
        }

        if self.aug_methods:
            self.augmentation(sample)

        sample = self._toTensor(sample)
        if self.label_format == 'rgb':
            sample["label"] = self._to_label_mask(sample["label"])
        else:
            sample['label'] = (sample['label'].squeeze(0) * 255).type(dtype=torch.long)

        return sample

    def _toTensor(self, sample):
        r"""
        Converts the incoming PIL image and ground truth image into 
        PyTorch tensors. 
        """
        sample['img']   = convert_pil_image_to_pytorch_tensor(sample['img'])
        sample['label'] = convert_pil_image_to_pytorch_tensor(sample['label'])
        return sample

    def _load_image(self, img_path: str):
        r""" Loading an image using PIL. """
        return Image.open(img_path)

    def _load_index_label(self, label_path: str):
        r""" Loads the annotation file in 8 bit format. """
        return Image.open(label_path).convert("L") # 8-bit pixels

    def _load_label_rgb(self, label_path: str):
        r""" Loads the annotation file in RGB format. """
        return Image.open(label_path).convert("RGB")

    def _to_label_mask(self, label_tensor: torch.Tensor):
        r"""
        Transforms RGB semantic segmentation label into index annotation.
        """
        mask_tensor = torch.ones(label_tensor.size(1), label_tensor.size(2), dtype=torch.long) * self.ignore_label
        
        rgb_tensor_int = torch.round(label_tensor * 255)
        for i_rgb, i_trainId in self.color2label.items():
            idx = (rgb_tensor_int == torch.tensor(i_rgb, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            valid_indices = (idx.sum(0) == 3)
            mask_tensor[valid_indices] = torch.tensor(i_trainId, dtype=torch.long)

        return mask_tensor

    def search_files(self):
        r"""
        Searches in a predefined path for the a2d2 dataset and collect
        all information. It requires the standard a2d2 folder structure.

        The list `filenames` contains tuples with four possible elements:

            * img: Path to RGB image file
            * lab: Path to RGB annotation file (optional)
            * lab_idx: Path to index annotation file (optional)
            * lab_idx_cs: Path to index annotation file following the cityscape format (optional)

        Each tuples requires at least one label file.
        """
        ## Using `cam_positions` only a subset (specific cameras only) of 
        ## the dataset can be used.
        try:
            self.cam_positions = self.cfg.cam_positions
        except omegaconf.errors.ConfigAttributeError:
            self.cam_positions = [
                'cam_front_center', 
                'cam_front_left', 
                'cam_front_right',
                'cam_side_left',
                'cam_side_right',
                'cam_rear_center'
            ]

        log_info("Searching for images of {}".format(self.cam_positions))

        self.filenames.clear()
        for folder in os.listdir(self.cfg.root):
            if folder.startswith("2018"):
                available_cams = os.listdir(os.path.join(self.cfg.root, folder, "camera"))
                for _cam in available_cams:
                    if _cam in self.cam_positions:
                        images = list(os.listdir(os.path.join(self.cfg.root, folder, "camera", _cam)))
                        images = [img for img in images if img.endswith(".png")]

                        for img in images:
                            paths_tuple = ()
                            img_path = os.path.join(self.cfg.root, folder, "camera", _cam, img)
                            if os.path.isfile(img_path):
                                paths_tuple = paths_tuple + (img_path,)
                            else:
                                continue

                            lab_path = os.path.join(self.cfg.root, folder, "label",  _cam, img.replace("camera", "label"))
                            if os.path.isfile(lab_path):
                                paths_tuple = paths_tuple + (lab_path,)
                            else:
                                paths_tuple = paths_tuple + (None,)

                            lab_index_path = os.path.join(self.cfg.root, folder, "label", _cam, lab_path.replace(".png", "_label_index.png"))
                            if os.path.isfile(lab_index_path):
                                paths_tuple = paths_tuple + (lab_index_path,)
                            else:
                                paths_tuple = paths_tuple + (None,)

                            lab_index_cityscape_path = os.path.join(self.cfg.root, folder, "label", _cam, lab_path.replace(".png", "_label_index_cityscape_format.png"))
                            if os.path.isfile(lab_index_cityscape_path):
                                paths_tuple = paths_tuple + (lab_index_cityscape_path,)
                            else:
                                paths_tuple = paths_tuple + (None,)

                            if len(paths_tuple) > 1:
                                self.filenames.append(paths_tuple)
        log_info("Found {} entries in A2D2 dataset.".format(
            len(self.filenames)
        ))

    @property
    def label2color(self):
        r""" Returns the mapping from indices to RGB format. """
        color = []
        cityscape_colors = [
            [128,  64, 128], [244,  35, 232], [ 70,  70,  70], 
            [102, 102, 156], [190, 153, 153], [153, 153, 153], 
            [250, 170,  30], [220, 220,   0], [107, 142,  35], 
            [152, 251, 152], [ 70, 130, 180], [255,   0,   0], 
            [220,  20,  60], [  0,   0, 142], [  0,   0,  70], 
            [  0,  60, 100], [  0,  80, 100], [  0,   0, 230], 
            [119,  11,  32]
        ]

        tmp   = {k:v for v, k in self.color2label.items() if k != self.ignore_label}
        tmp[self.ignore_label] = (0, 0, 0)
        for n in range(max(list(self.color2label.values())) + 1):
            try:
                if self.use_cityscape_format:
                    color.append(cityscape_colors[n])
                else:
                    color.append(list(tmp[n]))
            except (KeyError, IndexError):
                color.append([0, 0, 0])
        return np.asarray(color)

    @property
    def color2label(self):
        r""" Returns mapping from RGB to indices. """
        if self.use_cityscape_format:
            return  {   (204, 255, 153) : self.ignore_label, (182, 89, 6)    : 18, (150, 50, 4)    : 18,
                        (90, 30, 1)     : 18, (90, 30, 30)    : 18, (96, 69, 143)   : self.ignore_label, 
                        (241, 230, 255) :  2, (255, 0, 0)     : 13, (200, 0, 0)     : 13,
                        (150, 0, 0)     : 13, (128, 0, 0)     : 13, (128, 128, 0)   :  1, 
                        (128, 0, 255)   :  0, (180, 50, 180)  :  0, (72, 209, 204)  : self.ignore_label,
                        (255, 70, 185)  : self.ignore_label, (238, 162, 173) :  4, (64, 0, 64)     : self.ignore_label,
                        (147, 253, 194) :  8, (139, 99, 108)  : self.ignore_label, (255, 0, 128)   : self.ignore_label,
                        (200, 125, 210) :  0, (150, 150, 200) : self.ignore_label, (204, 153, 255) : 11, 
                        (189, 73, 155)  : 11, (239, 89, 191)  : 11, (255, 246, 143) :  5, 
                        (255, 0, 255)   :  0, (150, 0, 150)   :  0, (53, 46, 82)    : self.ignore_label, 
                        (185, 122, 87)  :  4, (233, 100, 0)   :  7, (180, 150, 200) :  1, 
                        (33, 44, 177)   :  6, (135, 206, 255) : 10, (238, 233, 191) :  0, 
                        (0, 255, 0)     : 17, (0, 200, 0)     : 17, (0, 150, 0)     : 17, 
                        (255, 193, 37)  :  0, (110, 110, 0)   :  0, (0, 0, 100)     : self.ignore_label, 
                        (159, 121, 238) :  7, (0, 255, 255)   :  7, (30, 220, 220)  :  7, 
                        (60, 157, 199)  :  7, (0, 128, 255)   :  6, (30, 28, 158)   :  6, 
                        (60, 28, 100)   :  6, (255, 128, 0)   : 14, (200, 128, 0)   : 14, 
                        (150, 128, 0)   : 14, (255, 255, 0)   : self.ignore_label, (255, 255, 200) : self.ignore_label,
                        (210, 50, 115)  :  0, (0, 0, 0)       : self.ignore_label }
        else:
            return {    (204, 255, 153) :  0, (182, 89, 6)    :  1, (150, 50, 4)    :  2,
                        (90, 30, 1)     :  3, (90, 30, 30)    :  4, (96, 69, 143)   :  5, 
                        (241, 230, 255) :  6, (255, 0, 0)     :  7, (200, 0, 0)     :  8,
                        (150, 0, 0)     :  9, (128, 0, 0)     : 10, (128, 128, 0)   : 11, 
                        (128, 0, 255)   : 12, (180, 50, 180)  : 13, (72, 209, 204)  : 14,
                        (255, 70, 185)  : 15, (238, 162, 173) : 16, (64, 0, 64)     : 17,
                        (147, 253, 194) : 18, (139, 99, 108)  : 19, (255, 0, 128)   : 20,
                        (200, 125, 210) : 21, (150, 150, 200) : 22, (204, 153, 255) : 23, 
                        (189, 73, 155)  : 24, (239, 89, 191)  : 25, (255, 246, 143) : 26, 
                        (255, 0, 255)   : 27, (150, 0, 150)   : 28, (53, 46, 82)    : 29, 
                        (185, 122, 87)  : 30, (233, 100, 0)   : 31, (180, 150, 200) : 32, 
                        (33, 44, 177)   : 33, (135, 206, 255) : 34, (238, 233, 191) : 35, 
                        (0, 255, 0)     : 36, (0, 200, 0)     : 37, (0, 150, 0)     : 38, 
                        (255, 193, 37)  : 39, (110, 110, 0)   : 40, (0, 0, 100)     : 41, 
                        (159, 121, 238) : 42, (0, 255, 255)   : 43, (30, 220, 220)  : 44, 
                        (60, 157, 199)  : 45, (0, 128, 255)   : 46, (30, 28, 158)   : 47, 
                        (60, 28, 100)   : 48, (255, 128, 0)   : 49, (200, 128, 0)   : 50, 
                        (150, 128, 0)   : 51, (255, 255, 0)   : 52, (255, 255, 200) : 53,
                        (210, 50, 115)  : 54, (0, 0, 0)       : self.ignore_label }

    def labelTensor2colorTensor(self, lab_tensor: torch.Tensor):
        r""" 
        Converts an index tensor of shape `[H, W]` into the RGB format
        using label2color() with a shape of `[C, H, W]`.
        """
        assert len(lab_tensor.shape) == 2, "Label tensor must be of shape [H, W]"
        return torch.from_numpy(self.label2color[lab_tensor]).permute(2, 0, 1).to(dtype=torch.uint8)

    def augmentation(self, sample):
        r""" 
        Method to apply data augmentation on a given sample.
        """
        for aug_method in self.aug_methods:
            if random() > 0.5:
                continue
            if aug_method == AUG_TYPE.BLUR:
                sample['img'] = blur_pil_image(sample['img'])
            elif aug_method == AUG_TYPE.RAIN:
                sample['img'] = addRain_pil_image(sample['img'])
            elif aug_method == AUG_TYPE.CLOUD:
                sample['img'] = addClouds_pil_image(sample['img'])
            elif aug_method == AUG_TYPE.FLIP_H:
                sample['img'] = hFlip_pil_image(sample['img'])
                sample['label'] = hFlip_pil_image(sample['label'])
            elif aug_method == AUG_TYPE.FLIP_V:
                sample['img'] = vFlip_pil_image(sample['img'])
                sample['label'] = vFlip_pil_image(sample['label'])
            else:
                pass




## Local tests
if __name__ == '__main__':
    project_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    cfg = omegaconf.DictConfig({'dataset' : {
        'name' : 'A2D2_ds',
        'root' : os.path.join(project_path, "data/a2d2/camera_lidar_semantic"),
        'batch_size' : 4,
        'validation_ratio' : 0,
        'use_cityscape_format' : True,
        'cam_positions' : [
            'cam_front_center', 
            'cam_front_left', 
            'cam_front_right'
            ],
        'label_format' : 'index_cityscape'
    }})
    a2d2_ds = A2D2_ds(cfg.dataset)
    a2d2_ds.split_train_valid()
    dataloader_training, _ = create_dataloader(cfg=cfg, ds1=a2d2_ds)
    for idx, batch in enumerate(dataloader_training):
        visualize_pytorch_tensor(
            py_tensor=batch['img'][0], 
            window_str="rgb img", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=a2d2_ds.labelTensor2colorTensor(batch['label'][0]),
            window_str="cityscape rgb label", 
            window_wait=0
        )
