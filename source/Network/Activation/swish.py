import torch
import torch.nn as nn
from typing import Optional


class Swish(nn.Module):
    r"""
    # Swish

    Definition of activation function can be found in original paper 'Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning'.

    Using the actual PyTorch implementation `nn.SiLU()` is not possible because it is not supported for the ONNX export yet.
    """
    __constants__ = ['inplace']
    inplace: bool
 
    def __init__(self, inplace: Optional[bool] = False):
        r"""
        Args:
        
        * `inplace (bool)`: 
            * Optional parameter to choose the inplace variant.
        """
        super().__init__()
        self.inplace = inplace
 
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r""" Forward path to process the activation function. """
        return input * torch.sigmoid(input)
 