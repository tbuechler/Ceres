import torch
import torch.nn.functional as F

from source.Logger.logger import *

def pytorch_tensor_OneHot(tensor: torch.Tensor, num_classes: int=None, ignore_idx: int=None):
    r"""
    Converts a given PyTorch Tensor into OneHot presentation.

    Args:

    * `tensor (torch.Tensor)`: 
        * Class index tensor of shape (N, H, W). 
    * `num_classes (int)`: 
        * Number of classes to consider. 
    * `ignore_idx (int)`: 
        * Usually the highest value in the label data and higher than num_classes. All elements equal to ignore_idx will be represented as [0, 0, 0, ..., 0] in One-Hot. 
    """
    if len(tensor.shape) == 4 and tensor.shape[1] == 1:
        tensor = torch.squeeze(tensor, dim=1)

    if len(tensor.shape) != 3:
        log_error("[TensorOneHot] Unexpected length of tensor shape {}. Expected a three-dimensional tensor with batch.".format(len(tensor.shape)))
        exit(-1)

    if ignore_idx is not None:
        ## Check that ignore_idx is greater than the number of classes. 
        assert ignore_idx > num_classes
        t_oneHot = F.one_hot(tensor, num_classes=ignore_idx+1).permute((0, 3, 1, 2))  # N, H, W, C -> N, C, H, W
        ## Cutoff everything above the number of classes at channel dimension to force all entries with ignore_idx to be 0.
        t_oneHot = t_oneHot[:, :num_classes, :, :]
    else:
        t_oneHot = F.one_hot(tensor, num_classes=num_classes).permute((0, 3, 1, 2))  # N, H, W, C -> N, C, H, W

    return t_oneHot