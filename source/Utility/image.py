import torch
import torchvision
import numpy as np
import imgaug.augmenters as iaa

from typing import Tuple
from source.Logger.logger import *
from PIL import Image, ImageFilter
from random import randint, uniform, randrange


def crop_pil_images_randomly(size: Tuple, *pil_imgs: Image) -> Tuple:
    r""" 
    Crops a variable number of PIL images in the same way.
    
    Args:

    * `size (Tuple)`: 
        * Tuple containing the cropped dimension: (height x width). 
    * `torch_tensors (Iterator(torch.Tensor))`: 
        * Iterator of variable number of PIL images. 
    """

    # Precheck: length of input
    if len(pil_imgs) < 1:
        return ()
    
    # Precheck: crop area does not exceed image dimension
    x_img, y_img = pil_imgs[0].size
    if (y_img < size[0]) or (x_img < size[1]):
        log_warning("[crop_pil_images_randomly] Crop size exceeds image dimension.")
        return pil_imgs

    # Precheck: all input images must have equal dimensions
    for pil_image in pil_imgs:
        x_tmp, y_tmp = pil_image.size
        if x_tmp != x_img or y_tmp != y_img:
            log_warning("[crop_pil_images_randomly] Image dimension of input must be equal for each element.")
            return pil_imgs

    # Compute random points for cropping
    crop_x = randrange(0, x_img - size[1])
    crop_y = randrange(0, y_img - size[0])

    toReturn = ()
    for pil_img in pil_imgs:
        toReturn += (pil_img.crop((crop_x, crop_y, crop_x + size[1], crop_y + size[0])),)
    return toReturn if len(pil_imgs) > 1 else toReturn[0]

def crop_pil_image(pil_img: Image, p1: Tuple, p2: Tuple) -> Image:
    r""" 
    Crops out an rectangular part of the PIL image.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    * `p1 (Iterator(Tuple))`: 
        * Coordinates of the upper left corner of the cropped part. 
    * `p2 (Iterator(Tuple))`: 
        * Coordinates to the bottom right corner  of the cropped part. 
    """
    return pil_img.crop((p1[0], p1[1], p2[0], p2[1]))

def hFlip_pil_image(pil_img: Image) -> Image:
    r""" 
    Flips a PIL Image horizontally.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    return pil_img.transpose(Image.FLIP_TOP_BOTTOM)

def vFlip_pil_image(pil_img: Image) -> Image:
    r""" 
    Flips a PIL Image vertically.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    return pil_img.transpose(Image.FLIP_LEFT_RIGHT)

def addRain_pil_image(pil_img: Image) -> Image:
    r""" 
    Adds an effect to the image which is similar to rain.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    seq = iaa.Rain(drop_size=(0.1, uniform(0.0005, 0.1)))
    return Image.fromarray(seq(image=np.asarray(pil_img)))

def addClouds_pil_image(pil_img: Image) -> Image:
    r""" 
    Adds an effect to the image which is similar to clouds.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    seq = iaa.Clouds()
    return Image.fromarray(seq(image=np.asarray(pil_img)))

def blur_pil_image(pil_img: Image) -> Image:
    r""" 
    Blurs the image with a random severity.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    return pil_img.filter(ImageFilter.GaussianBlur(radius = randint(1, 4)))

def switchChannel_pil_image(pil_img: Image) -> Image:
    """ 
    Switches the channels of an image: b <-> r.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    """
    return Image.fromarray(np.array(pil_img)[:,:,::-1])

def convert_pil_image_to_pytorch_tensor(pil_img: Image, channel_check=True) -> torch.Tensor:
    r""" 
    Transforms a PIL Image to a PyTorch tensor of shape CxHxW. 
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    * `channel_check (bool)`: 
        * If true, all channels greater than three are cut off to keep only RGB, BGR, YUV, Gray etc. 
    """
    return torchvision.transforms.ToTensor()(pil_img)[:3, :, :] if channel_check else torchvision.transforms.ToTensor()(pil_img)

def switchUpDown_pil_image(pil_img: Image, y_line: int) -> Image:
    r"""
    Switches two parts (one above each other) of the PIL Image.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    * `y_line (int)`: 
        * Point in vertical direction where both parts will be separated. 
    """
    tmp = convert_pil_image_to_pytorch_tensor(pil_img)
    tmp = switchUpDown_pytorch_tensor(tmp, y_line)
    return convert_pytorch_tensor_to_pil_image(tmp)

def switchLeftRight_pil_image(pil_img: Image, x_line: int) -> Image:
    r"""
    Switches two adjacent parts of the tensor.
    
    Args:

    * `pil_img (PIL.Image)`: 
        * PIL image to process. 
    * `y_line (int)`: 
        * Point in horizontal direction where both parts will be separated. 
    """
    tmp = convert_pil_image_to_pytorch_tensor(pil_img)
    tmp = switchLeftRight_pytorch_tensor(tmp, x_line)
    return convert_pytorch_tensor_to_pil_image(tmp)

def crop_pytorch_tensor_randomly(size: Tuple, *torch_tensors: torch.Tensor) -> Tuple:
    """ 
    Crops a variable number of torch tensors in the same way.
    
    Args:

    * `size (Tuple)`: 
        * Tuple containing the cropped dimension: (height x width). 
    * `torch_tensors (Iterator(torch.Tensor))`: 
        * Iterator of variable number of torch tensors. 
    """
    assert len(torch_tensors[0].shape) == 3, "[crop_pytorch_tensor_randomly] Torch tensor is expected to have the shape of (C x H x W)."

    if len(torch_tensors) < 1:
        return ()
    
    y_img, x_img = torch_tensors[0].shape[1:]
    if (y_img < size[0]) or (x_img < size[1]):
        log_warning("[crop_pytorch_tensor_randomly] Crop size exceeds tensor dimension.")
        return torch_tensors

    for torch_tensor in torch_tensors:
        y_tmp, x_tmp = torch_tensors[0].shape[1:]
        if x_tmp != x_img or y_tmp != y_img:
            log_warning("[crop_pytorch_tensor_randomly] Image dimension of input must be equal for each element.")
            return torch_tensors

    crop_x = randrange(0, x_img - size[1])
    crop_y = randrange(0, y_img - size[0])

    toReturn = ()
    for torch_tensor in torch_tensors:
        toReturn += (torch_tensor[:, crop_y:crop_y+size[0], crop_x:crop_x+size[1]],)
    return toReturn if len(torch_tensors) > 1 else toReturn[0]

def crop_pytorch_tensor(py_tensor: torch.Tensor, p1: Tuple, p2: Tuple) -> torch.Tensor:
    r""" 
    Crops out an rectangular part of the pytorch tensor.
    
    Args:

    * `py_tensor (torch.Tensor)`: 
        * Tensor have the shape C x H x W. 
    * `p1 (Tuple)`: 
        * Coordinates of the upper left corner of the cropped part. 
    * `p2 (Tuple)`: 
        * Coordinates to the bottom right corner  of the cropped part. 
    """
    assert len(py_tensor.shape) == 3, "Tensor shape must have a length of 3 and the form: C x H x W."
    return py_tensor[:, p1[0]:p2[0], p1[1]:p2[1]]

def hFlip_pytorch_tensor(py_tensor: torch.Tensor) -> torch.Tensor:
    r""" 
    Flips a pytorch tensor horizontally.
    
    Args:

    * `py_tensor (torch.Tensor)`: 
        * Tensor having the shape C x H x W. 
    """
    assert len(py_tensor.shape) == 3, "Tensor shape must have a length of 3 and the form: C x H x W."
    return torch.flip(py_tensor, [1])

def vFlip_pytorch_tensor(py_tensor: torch.Tensor) -> torch.Tensor:
    r""" 
    Flips a pytorch tensor vertically.
    
    Args:

    * `py_tensor (torch.Tensor)`: 
        * Tensor having the shape C x H x W. 
    """
    assert len(py_tensor.shape) == 3, "Tensor shape must have a length of 3 and the form: C x H x W."
    return torch.flip(py_tensor, [2])

def convert_pytorch_tensor_to_pil_image(py_tensor: torch.Tensor) -> Image:
    r""" Transforms a PyTorch tensor of shape CxHxW to a PIL Image. """
    assert len(py_tensor.shape) == 3, "Tensor shape must have a length of 3 and the form: C x H x W."
    return torchvision.transforms.ToPILImage()(py_tensor)

def switchUpDown_pytorch_tensor(py_tensor: torch.Tensor, y_line: int) -> torch.Tensor:
    r"""
    Switches two parts (one above each other) of the tensor.
    
    Args:

    * `py_tensor (torch.Tensor)`: 
        * Tensor having the shape C x H x W. 
    * `y_line (int)`: 
        * Point in vertical direction where both parts will be separated. 
    """
    up   = py_tensor[:, 0:y_line, :]
    down = py_tensor[:, y_line:, :]
    res  = torch.cat((down, up), dim=1)
    # Make sure shape does not change
    assert py_tensor.shape == res.shape, "Shape has changed of adapted image."
    return res

def switchLeftRight_pytorch_tensor(py_tensor: torch.Tensor, x_line: int) -> torch.Tensor:
    r"""
    Switches two adjacent parts of the tensor.
    
    Args:

    * `py_tensor (torch.Tensor)`: 
        * Tensor having the shape C x H x W. 
    * `x_line (int)`: 
        * Point in horizontal direction where both parts will be separated. 
    """
    left  = py_tensor[:, :, 0:x_line]
    right = py_tensor[:, :, x_line:]
    res  = torch.cat((right, left), dim=2)
    # Make sure shape does not change
    assert py_tensor.shape == res.shape, "Shape has changed of adapted image."
    return res
