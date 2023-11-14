import torch
import numpy as np


def is_numpy(data):
    r""" Checks if data is a numpy array. """
    return isinstance(data, np.ndarray)

def is_tensor(data):
    r""" Checks if data is a torch tensor. """
    return type(data) == torch.Tensor

def is_tuple(data):
    r""" Checks if data is a tuple. """
    return isinstance(data, tuple)

def is_list(data):
    r""" Checks if data is a list. """
    return isinstance(data, list)

def is_dict(data):
    r""" Checks if data is a dictionary. """
    return isinstance(data, dict)

def is_str(data):
    r""" Checks if data is a string. """
    return isinstance(data, str)

def is_int(data):
    r""" Checks if data is an integer. """
    return isinstance(data, int)

def is_seq(data):
    r""" Checks if data is a list or tuple. """
    return is_tuple(data) or is_list(data)

def is_nan(data):
    r""" Checks if data is nan. """
    return data != data