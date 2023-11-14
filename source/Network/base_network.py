import torch
import torch.nn as nn
from typing import Any


class BaseNetwork(nn.Module):
    r""" 
    # Base Network
    
    Base class for all network architectures. 
    """
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        r""" Forward pass for each network. """
        raise NotImplementedError

    def load(self, checkpoint_path: str):
        r"""
        Loads checkpoint into network instance. 
        
        Args:
        
        * `checkpoint_path (str)`:
            * Path of the pretrained checkpoint.
        """
        self.load_state_dict(torch.load(checkpoint_path), strict=False)

    def init_weights(self):
        r""" Initialization of weight parameter. """
        raise NotImplementedError
    
    def export(self, input_args: Any, dir_path: str, file_name: str=None):
        r"""
        Export function to save network checkpoint or ONNX export. 
        
        Args:
        
        * `input_args (Tuple(torch.Tensor,))`:
            * Dummy input to make export available via tracing.
        * `dir_path (str)`:
            * Path where the checkpoints will be exported to.
        * `file_name (str)`:
            * If given, it will be the checkpoint name.
        """
        raise NotImplementedError
