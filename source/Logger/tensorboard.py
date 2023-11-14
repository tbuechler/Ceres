import torch
from source.Model.model_wrapper import ModelWrapper


def show_graph(
    writer          : torch.utils.tensorboard.SummaryWriter, 
    model_wrapper   : ModelWrapper, 
    input           : torch.Tensor) -> None:
    r""" Shows the used network architecture in TensorBoard
    
    Args:
    
    * `writer (torch.utils.tensorboard.SummaryWrite)`:
        * Tensorboard instance.
    * `model_wrapper (ModelWrapper)`:
        * ModelWrapper instance containing the used network architecture.
    * `input (torch.Tensor)`:
        * Torch Tensor to be fed into the graph. Shape depends on the network architecture.
    """
    writer.add_graph(model=model_wrapper.model_arch, input_to_model=input, verbose=False)

def show_image(
    writer      : torch.utils.tensorboard.SummaryWriter, 
    tag         : str, 
    image_tensor: torch.Tensor, 
    dataformats : str='CHW', 
    global_step : int=None) -> None:
    r""" Shows a single image in TensorBoard from a PyTorch tensor. 
    
    Args:
    
    * `writer (torch.utils.tensorboard.SummaryWrite)`:
        * Tensorboard instance.
    * `tag (str)`:
        * Tag used in tensorboard for the image.
    * `image_tensor (torch.Tensor)`:
        * Image torch tensor of shape `[C, H, W]`.
    * `global_step (int)`:
        * Step number at which the image was logged.    
    """
    assert len(image_tensor.shape) == 3, "show_image() can only be used with an input tensor of shape CxHxW, but length of input tensor is {}.".format(len(image_tensor.shape))
    writer.add_image(tag=tag, img_tensor=image_tensor, global_step=global_step, dataformats=dataformats)

def show_images(
    writer      : torch.utils.tensorboard.SummaryWriter, 
    tag         : str, 
    image_tensor: torch.Tensor, 
    global_step : int=None) -> None:
    r""" Shows a batch of images in TensorBoard from a PyTorch tensor. 
    
    Args:
    
    * `writer (torch.utils.tensorboard.SummaryWrite)`:
        * Tensorboard instance.
    * `tag (str)`:
        * Tag used in tensorboard for the image.
    * `image_tensor (torch.Tensor)`:
        * Image torch tensor of shape `[N, C, H, W]`.
    * `global_step (int)`:
        * Step number at which the image was logged.    
    """
    assert len(image_tensor.shape) == 4, "show_images() can only be used with an input tensor of shape NxCxHxW, but length of input tensor is {}.".format(len(image_tensor.shape))
    writer.add_images(tag=tag, img_tensor=image_tensor, global_step=global_step)

def show_scalar(
    writer      : torch.utils.tensorboard.SummaryWriter, 
    tag         : str, 
    data        : torch.Tensor, 
    global_step : int=None):
    r""" Shows the progress of one scalar value. 
    
    Args:
    
    * `writer (torch.utils.tensorboard.SummaryWrite)`:
        * Tensorboard instance.
    * `tag (str)`:
        * Tag used in tensorboard for the image.
    * `data (torch.Tensor)`:
        * Must be scalar value.
    * `global_step (int)`:
        * Step number at which the image was logged.    
    """    
    writer.add_scalar( tag = tag, scalar_value = data, global_step = global_step)

def show_scalars(
    writer      : torch.utils.tensorboard.SummaryWriter, 
    tag         : str, 
    global_step : int=None, 
    **data      : dict):
    r""" Shows the progress of multiple scalar values in one plot simultaneously. The variable names are used for indication. 
    
    Args:
    
    * `writer (torch.utils.tensorboard.SummaryWrite)`:
        * Tensorboard instance.
    * `tag (str)`:
        * Tag used in tensorboard for the image.
    * `global_step (int)`:
        * Step number at which the image was logged. 
    * `**data (dict)`:
        * Key-value pair storing the tag and corresponding value.   
    """   
    writer.add_scalars(
        main_tag        = tag,
        tag_scalar_dict = data,
        global_step     = global_step
    )
