import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(
        os.path.dirname(__file__), os.path.join("..", "..", "..")
    ))
    from source.Utility.visualizer import visualize_pytorch_tensor
    from source.Dataloader.Utility.converter import labImg2EdgeTorch
    from source.Dataloader.Utility.dataloader import create_dataloader

import cv2
import json
import torch
import omegaconf
import numpy as np

from PIL import Image
from source.Utility.image import *
from torchvision.transforms import transforms
from source.Utility.augmentation import AUG_TYPE
from source.Dataloader.datahandler import DataHandler
from source.Logger.logger import log_error, log_warning
from source.Utility.visualizer import visualize_pytorch_tensor


class Cityscape_ds(DataHandler):
    r"""
    # Cityscape Dataset

    Dataset for the Cityscape dataset (https://www.cityscapes-dataset.com).
    Currently it does support the semantic segmentation task along
    with the stereo setup. 
    """
    ignore_label  = 255
    class_weights = torch.FloatTensor([
        0.8373, 0.918, 0.866, 1.0345, 
        1.0166, 0.9969, 0.9754, 1.0489,
        0.8786, 1.0023, 0.9539, 0.9843, 
        1.1116, 0.9037, 1.0865, 1.0955, 
        1.0865, 1.1529, 1.0507
    ])

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        r"""    
        Args:
        
        * `cfg (omegaconf.DictConfig)`:
            * Hydra based configuration dictionary based on the given configuration .yaml file. **Only the subset cfg object for the dataset is available at this point.**
        """
        super().__init__(cfg)

        ## It is possible to indicate labels to be ignored by overwriting
        ## `ignore_label`. The default value for this parameter is 255.
        try:
            self.ignore_label = self.cfg.ignore_label
        except omegaconf.errors.ConfigAttributeError:
            pass

        ## Publications show that standardization usually perform better
        ## with the Cityscape dataset. So it is possible to provide
        ## the mean and standard deviation per channel for the input images.
        try:
            self.transforms = transforms.Compose([
                Standardize(mean=self.cfg.normalization_mean, std=self.cfg.normalization_std)
            ])
        except omegaconf.errors.ConfigAttributeError:
            self.transforms = None
            log_warning("Missing values for mean (normalization_mean) and std (normalization_std) for normalization process. Continue without transformation...")

    def __getitem__(self, index: int):
        r"""
        Creates a sample from the dataset for a specific pair of data 
        using an index. It will load the images using PIL, convert ground
        truth data if necessary, apply data augmentation and convert 
        sample entries into PyTorch Tensors. Additionally, it loads
        the disparity map as 8-bit single channel images.
                
        Returns:
            Dictionary sample in form of {'img_left' : img_left, 'img_right' : img_right, 'disparity' : disparity, 'label' : label, 'camera' : camera_conf }
        """

        img_left    = self._load_image(img_path=self.filenames[index][0])
        img_right   = self._load_image(img_path=self.filenames[index][1])
        disparity   = self._load_disparity(disparity_path=self.filenames[index][2])
        label       = self._load_label(label_path=self.filenames[index][3])
        camera_conf = self._load_camera_conf(camera_path=self.filenames[index][4]) 

        sample =  { 
            'img_left' : img_left, 'img_right' : img_right,
            'disparity' : disparity, 'label' : label,
            'camera' : camera_conf 
        }

        if self.aug_methods:
            self.augmentation(sample)
        sample = self._toTensor(sample)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _toTensor(self, sample):
        r"""
        Converts the incoming PIL images (left and right), the ground truth image and the disparity map into PyTorch tensors. 
        """
        sample['img_left']  =   convert_pil_image_to_pytorch_tensor(
                                    pil_img=sample['img_left']
                                )
        sample['img_right'] =   convert_pil_image_to_pytorch_tensor(
                                    sample['img_right']
                                )
        sample['disparity'] = torch.from_numpy(sample['disparity'])
        sample['label']     = torch.from_numpy(sample['label']).to(dtype=torch.long)
        return sample

    def _load_camera_conf(self, camera_path: str):
        r""" Loads the configuration file for the stereo camera setup. """
        with open(camera_path,'r') as f:
            cam_conf = json.load(f)
        return cam_conf

    def _load_disparity(self, disparity_path: str):
        r""" 
        Loading disparity map and converting to correct values according to the given documentation from Cityscape. 
        """
        disp = cv2.imread(disparity_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        disp[disp > 0] = (disp[disp > 0] - 1) / 256
        return disp

    def _load_image(self, img_path: str):
        r""" Load an image using PIL. """
        return Image.open(img_path)

    def _load_label(self, label_path: str):
        r""" Loads the annotation file and returns list of annotations. """
        lab = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        tmp = lab.copy()
        for k, v in self.label_mapping.items():
            lab[tmp == k] = v
        return lab

    def search_files(self):
        r"""
        Searches in a predefined path for the cityscape dataset and collect
        all information. It requires the standard cityscape folder structure.

        The list `filenames` contains tuples with five possible elements:

            * img_left: Path to RGB image file of left camera
            * img_right: Path to RGB image file of right camera
            * disparity: Path to disparity map file
            * label: Path to index gtFine_labelIds file
            * camera: Path to camera configuration file
        """
        self.filenames.clear()
        try:
            rootImgDir = os.path.join(self.cfg.root, 'leftImg8bit', self.cfg.set_name)
        except omegaconf.errors.ConfigAttributeError:
            log_error("Missing 'set_name' key [train, val or test] in configuration object for the CityScape dataset.")

        for folder in os.listdir(rootImgDir):
            for parent_path, _, filenames in os.walk(os.path.join(rootImgDir, folder)):
                for f in filenames:
                    if f.endswith(".png"):
                        fullImgLeftPath     = os.path.join(parent_path, f)
                        fullImgRightPath    = fullImgLeftPath.replace("leftImg8bit", "rightImg8bit")
                        fullDisparityPath   = fullImgLeftPath.replace("leftImg8bit", "disparity")
                        fullGtFinePath      = fullImgLeftPath.replace("leftImg8bit", "gtFine").replace("_gtFine", "_gtFine_labelIds")
                        fullCameraPath      = fullImgLeftPath.replace("leftImg8bit", "camera").replace(".png", ".json")
                        if all(os.path.isfile(x) for x in [fullImgLeftPath, fullImgRightPath, fullDisparityPath, fullGtFinePath, fullCameraPath]):
                            self.filenames.append(
                                (
                                    fullImgLeftPath, 
                                    fullImgRightPath,
                                    fullDisparityPath,
                                    fullGtFinePath,
                                    fullCameraPath
                                )
                            )
        log_info("Found {} entries in Cityscape dataset.".format(len(self.filenames)))

    @property
    def label_mapping(self):
        r""" Returns the mapping from given label id to common training id. """
        return {-1: self.ignore_label, 0: self.ignore_label, 
                1: self.ignore_label, 2: self.ignore_label, 
                3: self.ignore_label, 4: self.ignore_label, 
                5: self.ignore_label, 6: self.ignore_label, 
                7: 0, 8: 1, 9: self.ignore_label, 
                10: self.ignore_label, 11: 2, 12: 3, 
                13: 4, 14: self.ignore_label, 15: self.ignore_label, 
                16: self.ignore_label, 17: 5, 18: self.ignore_label, 
                19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                25: 12, 26: 13, 27: 14, 28: 15, 
                29: self.ignore_label, 30: self.ignore_label, 
                31: 16, 32: 17, 33: 18}
    
    @property
    def label2color(self):
        r""" Returns the mapping from training id to RGB color. """
        color = []
        tmp   = {k:v for v, k in self.color2label.items() if k != self.ignore_label}
        tmp[self.ignore_label] = (0, 0, 0)
        for n in range(max(list(self.color2label.values())) + 1):
            try:
                color.append(list(tmp[n]))
            except KeyError:
                color.append([0, 0, 0])
        return np.asarray(color)

    @property
    def color2label(self):
        r""" Returns the mapping from RGB color to training id. """
        return {(0,  0,  0)   : self.ignore_label, 
                (111, 74,  0) : self.ignore_label, 
                ( 81,  0, 81) : self.ignore_label, 
                (128, 64,128) :  0, 
                (244, 35,232) :  1, 
                (250,170,160) : self.ignore_label, 
                (230,150,140) : self.ignore_label, 
                ( 70, 70, 70) :  2, 
                (102,102,156) :  3, 
                (190,153,153) :  4,  
                (180,165,180) : self.ignore_label, 
                (150,100,100) : self.ignore_label, 
                (150,120, 90) : self.ignore_label, 
                (153,153,153) :  5, 
                (250,170, 30) :  6, 
                (220,220,  0) :  7, 
                (107,142, 35) :  8, 
                (152,251,152) :  9, 
                ( 70,130,180) : 10, 
                (220, 20, 60) : 11, 
                (255,  0,  0) : 12, 
                (  0,  0,142) : 13, 
                (  0,  0, 70) : 14, 
                (  0, 60,100) : 15, 
                (  0,  0, 90) : self.ignore_label, 
                (  0,  0,110) : self.ignore_label, 
                (  0, 80,100) : 16, 
                (  0,  0,230) : 17, 
                (119, 11, 32) : 18 }

    def labelTensor2colorTensor(self, lab_tensor: torch.Tensor):
        r""" 
        Converts a label tensor of shape `[H, W]` to a RGB tensor of shape `[C, H, W]`. 
        """
        assert len(lab_tensor.shape) == 2, "Label tensor must be of shape [H, W]"
        return torch.from_numpy(self.label2color[lab_tensor]).permute(2, 0, 1).to(dtype=torch.uint8)

    def augmentation(self, sample):
        r""" 
        Method to apply data augmentation on a given sample.
        """
        for aug_method in self.aug_methods:
            if aug_method == AUG_TYPE.BLUR:
                sample['img_left'] = blur_pil_image(sample['img_left'])
                sample['img_right'] = blur_pil_image(sample['img_right'])
            elif aug_method == AUG_TYPE.RAIN:
                sample['img_left'] = addRain_pil_image(sample['img_left'])
                sample['img_right'] = addRain_pil_image(sample['img_right'])
            elif aug_method == AUG_TYPE.CLOUD:
                sample['img_left'] = addClouds_pil_image(sample['img_left'])
                sample['img_right'] = addClouds_pil_image(sample['img_right'])
            else:
                pass

class Standardize(object):
    r""" 
    Transformation class to apply standardization, such that
    $$ image = (image - mean) / std.$$
    """
    def __init__(self, mean: list, std: list) -> None:
        r"""
        Args:

        * `mean (list)`:
            * List of mean values for each channel of current dataset.
        * `std (list)`:
            * List of std values for each channel of current dataset.
        """
        assert len(mean) > 0
        assert len(std) > 0

        self.mean = torch.tensor([[[mean[0]]], [[mean[1]]], [[mean[2]]]])
        self.std  = torch.tensor([[[std[0]]], [[std[1]]], [[std[2]]]])

    def __call__(self, sample):
        r""" Applies the standardization to the input sample. """
        sample['img_left'] = (sample['img_left'] - self.mean) / self.std
        sample['img_right'] = (sample['img_right'] - self.mean) / self.std
        return sample




## Local tests
if __name__ == '__main__':
    project_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    cfg = omegaconf.DictConfig({'dataset' : {
        'name' : 'Cityscape_ds',
        'root' : os.path.join(project_path, "data/cityscape"),
        'set_name' : 'train',
        'augmentation_methods' : ['blur'],
        'batch_size' : 2,
        'validation_ratio' : 0,
        'normalization_mean': [0.485, 0.456, 0.406],
        'normalization_std': [0.229, 0.224, 0.225]
    }})
    cityscape_ds = Cityscape_ds(cfg.dataset)
    cityscape_ds.split_train_valid()
    dataloader_training, _ = create_dataloader(cfg=cfg, ds1=cityscape_ds)
    for idx, batch in enumerate(dataloader_training):
        edge = torch.stack([labImg2EdgeTorch(batch['label'][i]) for i in range(batch['label'].shape[0])])
        visualize_pytorch_tensor(
            py_tensor=edge[0], 
            window_str="edge", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['img_left'][0],
            window_str="left", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['img_right'][0], 
            window_str="right", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=cityscape_ds.labelTensor2colorTensor(batch['label'][0]),
            window_str="label", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['disparity'][0].type(torch.uint8), 
            window_str="disparity", 
            window_wait=0
        )

