import os
if __name__ == '__main__':
    import sys
    sys.path.append(os.path.join(
        os.path.dirname(__file__), os.path.join("..", "..", "..")
    ))
    from source.Dataloader.Utility.dataloader import create_dataloader
    from source.Utility.ObjectDetection2D.bounding_box2D import BBox2D, visualize

import torch
import omegaconf
import numpy as np
import torchvision.transforms as T

from PIL import Image
from typing import Tuple
from source.Utility.image import *
from torchvision.transforms import transforms
from source.Utility.augmentation import AUG_TYPE
from source.Dataloader.datahandler import DataHandler


class Kitti2D_ds(DataHandler):
    r"""
    # Kitti Dataset

    Dataset for the Kitti dataset (https://www.cvlibs.net/datasets/kitti/).
    Currently it does only support the task of 2D Object Detection and thus
    will only provide an image and the corresponding annotation file for the
    classified objects.
    """
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

        ## List of classes for the classification task that are supported
        ## by the Kitti dataset.
        self.classes = [
            'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person_sitting', 'Cyclist', 'Tram'
        ]

    def __getitem__(self, index: int):
        r"""
        Creates a sample from the dataset for a specific pair of data 
        using an index. It will load the image using PIL and the annotation file, 
        apply data augmentation and convert sample entries into PyTorch Tensors.
                
        Returns:
            Dictionary sample in form of {'img' : img, 'annotation' : annotation}.
        """
        img = self._load_image(img_path=self.filenames[index][0])
        ano = self._load_annotation(label_path=self.filenames[index][1])
        
        sample = { 'img' : img, 'annotation' : ano }
        if self.aug_methods:
            self.augmentation(sample)
        sample = self._toTensor(sample)
        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _load_image(self, img_path: str):
        r""" Loading an image using PIL. """
        return Image.open(img_path)

    def _load_annotation(self, label_path: str):
        r""" 
        Loads the annotation file and returns list of annotations. The annotations are in form of 
        $$(x_l, y_t, x_r, y_b, class).$$
        """
        with open(label_path) as f:
            content = f.readlines()
        content = [x.split() for x in content]

        annotations = np.zeros((0, 5))
        for c in content:
            if (c[0] in self.classes):
                bbox = np.array(c[4:8], dtype = "float32")
                annotation = np.zeros((1, 5))
                annotation[0, :4] = bbox
                annotation[0, 4]  = self.classes.index(c[0])
                annotations = np.append(annotations, annotation, axis=0)
        return annotations

    def search_files(self):
        r"""
        Searches in a predefined path for the kitti dataset and collect
        all information. It requires the standard kitti folder structure.

        The list `filenames` contains tuples with two possible elements:

            * img: Path to RGB image file
            * anno: Path to the annotation file
        """        
        imgs_path = os.path.join(self.root, "images/training/image_2")
        anos_path = os.path.join(self.root, "labels/training/label_2")

        self.filenames = []
        for img_name in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_name)
            if os.path.isfile(img_path) and img_name.endswith(".png"):
                ano_path = os.path.join(anos_path, img_name.replace(".png", ".txt"))
                if os.path.isfile(ano_path):
                    self.filenames.append((img_path, ano_path))

    def augmentation(self, sample):
        r""" 
        Method to apply data augmentation on a given sample.
        """
        for aug_method in self.aug_methods:
            if aug_method == AUG_TYPE.BLUR:
                sample['img'] = blur_pil_image(sample['img'])
            elif aug_method == AUG_TYPE.RAIN:
                sample['img'] = addRain_pil_image(sample['img'])
            elif aug_method == AUG_TYPE.CLOUD:
                sample['img'] = addClouds_pil_image(sample['img'])
            else:
                pass

    def _toTensor(self, sample):
        r"""
        Converts the incoming PIL image and numpy array of the image and annotation into PyTorch tensors. 
        """
        sample['img']  = convert_pil_image_to_pytorch_tensor(sample['img'])
        sample['annotation'] = torch.from_numpy(sample['annotation'])
        return sample

    @staticmethod
    def collate_fn(samples: dict):
        r"""
        This custom overload of the collate_fn is necessary and must be passed to the Dataloader
        to handle variable size of bounding boxes per image in one batch.
        
        Args:

        * `samples (list[dict])`:
            * List of samples (elements of one batch) containing the image and the corresponding
            annotation information of it.

        The collate_fn receives a list of tuples if your __getitem__ function from a Dataset
        subclass returns a tuple, or just a normal list if your Dataset subclass returns only one
        element. Its main objective is to create your batch without spending much time implementing
        it manually. Try to see it as a glue that you specify the way examples stick together in a
        batch. If you don’t use it, PyTorch only put batch_size examples together as you would
        using torch.stack (not exactly it, but it is simple like that).
        """
        imgs  = [sample['img'] for sample in samples]
        annos = [sample['annotation'] for sample in samples]

        ## Transformation of the shape of the images
        ## $[(C, H, W), ...., (C, H, W)] \rightarrow (B, C, H, W)$
        imgs = torch.stack(imgs, axis=0)

        ## Get the maximum number of bounding boxes for the incoming list of annotation to 
        ## create a batch with a constant number of bounding boxes.
        max_anno = max(anno.shape[0] for anno in annos)

        ## If the number of bounding boxes is greater than zero fill up a new tensor. Annotations
        ## which have been "padded" with new bounding boxes to get a constant bounding box number 
        ## are annotated with -1 and are invalid.
        if max_anno > 0:
            new_annos = torch.ones((len(annos), max_anno, 5)) * -1
            for idx, anno in enumerate(annos):
                if anno.shape[0] > 0:
                    new_annos[idx, :anno.shape[0], :] = anno

        ## If no bounding boxes were found in the current batch, which can happen if the image has 
        ## nothing to detect, than the tensor is filled with invalid bounding boxes, indicated by
        ## the value -1.
        else:
            new_annos = torch.ones((len(annos), 1, 5)) * -1

        ## Return structure after the custom collate_fn.
        return {
            'img'  : imgs,
            'annotation' : new_annos
        }

class Resize(object):
    r"""
    Transformation instance to resize image and adapt annotations accordingly.
    """
    def __init__(self, img_size: Tuple[int, int]) -> None:
        r"""
        Args:

        * `img_size (Tuple[int, int])`:
            * Size of image size resizing to [height, width].
        """
        self.height      = img_size[0]
        self.width       = img_size[1]
        self.t_resize    = T.Resize(size = (self.height, self.width), antialias=True)

    def __call__(self, sample: dict):
        r""" 
        Args:
        
        * `sample (dict)`:
            * Dictionary of one sample containing information about the image and the corresponding annotation.s
        """
        img, anno                   = sample['img'], sample['annotation']
        _, height_old, width_old    = img.shape
        scale_height                = self.height / height_old
        scale_width                 = self.width / width_old

        img = self.t_resize(img)

        ## The pixel coordinates [0:4] are scaled according to the change in image height and image width.
        anno[:, 0:4:2] *= scale_width
        anno[:, 1:4:2] *= scale_height
        
        ## Return structure after resizing.
        return {
            'img'  : img,
            'annotation' : anno
        }




## Local tests
if __name__ == '__main__':
    project_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    cfg = omegaconf.DictConfig({'dataset' : {
        'name' : 'Kitti2D_ds',
        'root' : os.path.join(project_path, "data/kitti2D"),
        'augmentation_methods' : ['rain', 'cloud', 'blur'],
        'batch_size' : 1,
        'validation_ratio' : 0,
        'image_width' : 1280,
        'image_height' : 384
    }})
    kitti2d_ds = Kitti2D_ds(cfg.dataset)
    kitti2d_ds.split_train_valid()
    dataloader_training, _ = create_dataloader(cfg=cfg, ds1=kitti2d_ds)
    for idx, batch in enumerate(dataloader_training):
        bbox = []
        for i_box in range(batch['annotation'][0].shape[0]):
            if int(batch['annotation'][0][i_box][4]) < 0: # Skip invalid bb's
                continue
            bbox.append(
                BBox2D(
                    int(batch['annotation'][0][i_box][0]), 
                    int(batch['annotation'][0][i_box][1]), 
                    int(batch['annotation'][0][i_box][2]), 
                    int(batch['annotation'][0][i_box][3]), 
                    int(batch['annotation'][0][i_box][4]), 1.0
                )
            )
        visualize(
            img=batch['img'][0], 
            boxes=bbox, 
            imshow=True, 
            waitKey=0, 
            objectList=kitti2d_ds.classes
        )
   
