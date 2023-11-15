import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(
        os.path.dirname(__file__), os.path.join("..", "..", "..")
    ))
    from source.Utility.visualizer import *
    from source.Dataloader.Utility.dataloader import create_dataloader

import torch
import omegaconf
import numpy as np
import torchvision.transforms as T

from PIL import Image
from source.Utility.image import *
from torchvision.transforms import transforms
from source.Utility.augmentation import AUG_TYPE
from source.Dataloader.datahandler import DataHandler


class SynthiaSF_ds(DataHandler):
    r"""
    # SynthiaSF Dataset

    Dataset for the Synthia-SF dataset (https://synthia-dataset.net).
    Currently it does only support the task of Stereo 3D-Reconstruction and thus will only 
    provide the left and right stereo image and the depth/disparity maps for each of these cameras.
    """
    ## Extrinsic parameters for the stereo setup (Coming from the 
    ## corresponding paper).
    baseline    = 0.6
    focallength = 847.630211643

    def __init__(self, cfg: omegaconf.DictConfig) -> None:
        r"""    
        Args:
        
        * `cfg (omegaconf.DictConfig)`:
            * Hydra based configuration dictionary based on the given configuration .yaml file.
            **Only the subset cfg object for the dataset is available at this point.**
        """
        super().__init__(cfg)

        self.transforms = transforms.Compose([
            Resize(img_size=(cfg.image_height, cfg.image_width))
        ])

    def __getitem__(self, index: int):
        r"""
        Creates a sample from the dataset for a specific pair of data 
        using an index. It will load both images using PIL and the raw depth map. The depth map
        will be converted into an useable format and disparity maps will be created out of it in
        combination with the extrinsic parameters. Furthermore, possible data augmentation and the
        transformation into PyTorch Tensors will be done.
                
        Returns:
            Dictionary sample in form of { 'img_left' : img_left, 'img_right' : img_right,
            'depth_left' : depth_left, 'depth_right' : depth_right, 'disp_left' : disp_left,
            'disp_right' : disp_right }.
        """
        img_left    = self._load_image(img_path=self.filenames[index][0])
        img_right   = self._load_image(img_path=self.filenames[index][1])
        depth_left  = self._load_depth(depth_path=self.filenames[index][2])
        depth_left[depth_left == 0.0] = 0.1
        depth_right = self._load_depth(depth_path=self.filenames[index][3])    
        depth_right[depth_right == 0.0] = 0.1    
        disp_left   = (self.baseline * self.focallength) / depth_left
        disp_right  = (self.baseline * self.focallength) / depth_right

        sample =  { 
            'img_left'      : img_left,   'img_right'   : img_right,
            'depth_left'    : depth_left, 'depth_right' : depth_right,
            'disp_left'     : disp_left,  'disp_right'  : disp_right
        }

        sample = self._toTensor(sample)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _toTensor(self, sample):
        r"""
        Converts the incoming PIL images and numpy arrays of the image, the depth map and the disparity maps into PyTorch tensors. 
        """
        sample['img_left']      = convert_pil_image_to_pytorch_tensor(sample['img_left'])
        sample['img_right']     = convert_pil_image_to_pytorch_tensor(sample['img_right'])
        sample['depth_left']    = torch.from_numpy(sample['depth_left'])
        sample['depth_right']   = torch.from_numpy(sample['depth_right'])
        sample['disp_left']     = torch.from_numpy(sample['disp_left'])
        sample['disp_right']    = torch.from_numpy(sample['disp_right'])
        return sample

    def _load_depth(self, depth_path: str):
        r""" 
        Loads the given three channel depth map that is furthermore converted according to the
        documentation to achieve an one channel depth map. 
        """
        img          = np.asarray(Image.open(depth_path))
        depth_matrix = (img[:, :, 0] + (img[:, :, 1] * 256.) + (img[:, :, 2] * 256. * 256.)) / ((256. * 256. * 256.) - 1.) * 1000.
        return depth_matrix

    def _load_image(self, img_path: str):
        r""" Loading an image using PIL. """
        return Image.open(img_path)

    def search_files(self):
        r"""
        Searches in a predefined path for the Synthia-SF dataset and collect
        all information. It requires the standard Synthia-SF folder structure.

        The list `filenames` contains tuples with two possible elements:

            * img_left: Path to the RGB image file of the left camera
            * img_right: Path to the RGB image file of the right camera
            * depth_left: Path to the depth map file for the left camera
            * depth_right: Path to the depth map file for the right camera
        """        
        self.filenames.clear()
        for folder in os.listdir(self.cfg.root):
            if not folder.startswith('SEQ'):
                continue
            leftImgPath = os.path.join(self.cfg.root, folder, 'RGBLeft')
            for parent_path, _, filenames in os.walk(leftImgPath):
                for f in filenames:
                    if f.endswith(".png"):
                        fullImgLeftPath     = os.path.join(parent_path, f)
                        fullImgRightPath    = fullImgLeftPath.replace("RGBLeft", "RGBRight")
                        fullDepthLeftPath   = fullImgLeftPath.replace("RGBLeft", "DepthLeft")
                        fullDepthRightPath  = fullImgLeftPath.replace("RGBLeft", "DepthRight")
                        if all(os.path.isfile(x) for x in [fullImgLeftPath, fullImgRightPath, fullDepthLeftPath, fullDepthRightPath]):
                            self.filenames.append(
                                (
                                    fullImgLeftPath, 
                                    fullImgRightPath,
                                    fullDepthLeftPath,
                                    fullDepthRightPath
                                )
                            )

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

class Resize(object):
    r""" Transformation instance to resize images. """
    def __init__(self, img_size: Tuple[int, int]) -> None:
        r"""
        Args:

        * `img_size (Tuple[int, int])`:
            * Size of image size resizing to [height, width].
        """
        self.height      = img_size[0]
        self.width       = img_size[1]
        self.t_resize    = T.Resize(
            size=(self.height, self.width), 
            antialias=True
        )

    def __call__(self, sample: dict):
        r""" Applies the resize operation to the input sample. """
        return {
            'img_left'      : self.t_resize(sample['img_left']),
            'img_right'     : self.t_resize(sample['img_right']),
            'depth_left'    : self.t_resize(sample['depth_left'].unsqueeze_(0)),
            'depth_right'   : self.t_resize(sample['depth_right'].unsqueeze_(0)),
            'disp_left'     : self.t_resize(sample['disp_left'].unsqueeze_(0)),
            'disp_right'    : self.t_resize(sample['disp_right'].unsqueeze_(0))
        }




## Local tests
if __name__ == '__main__':
    project_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    cfg = omegaconf.DictConfig({'dataset' : {
        'name' : 'SynthiaSF_ds',
        'root' : os.path.join(project_path, "data/synthiaSF"),
        'batch_size' : 2,
        'validation_ratio' : 0,
        'image_width' : 2048,
        'image_height' : 1024
    }})
    synthiaSF_ds = SynthiaSF_ds(cfg.dataset)
    synthiaSF_ds.split_train_valid()
    dataloader_training, _ = create_dataloader(cfg=cfg, ds1=synthiaSF_ds)
    for idx, batch in enumerate(dataloader_training):
        visualize_overlay_pytorch_tensors(
            py_tensor_1=batch['img_left'][0],
            alpha_1=0.5, 
            py_tensor_2=batch['img_right'][0], 
            alpha_2=0.5, 
            window_str="weighted", 
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
            py_tensor=batch['depth_left'][0].type(torch.uint8), 
            window_str="depth_left", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['depth_right'][0].type(torch.uint8),
            window_str="depth_right", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['disp_left'][0].type(torch.uint8), 
            window_str="disp_left", 
            window_wait=1
        )
        visualize_pytorch_tensor(
            py_tensor=batch['disp_right'][0].type(torch.uint8), 
            window_str="disp_right", 
            window_wait=0
        )
