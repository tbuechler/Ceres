import cv2
import torch
import numpy as np

from PIL import Image


def visualize_overlay_pytorch_tensors(
    py_tensor_1: torch.Tensor,
    alpha_1: float,
    py_tensor_2: torch.Tensor,
    alpha_2: float,
    window_str: str="",
    window_wait: int=1):
    r""" Visualize two PyTorch tensor on top of each other. """
    assert py_tensor_1.shape == py_tensor_2.shape, "Shape of both tensor must be the same."
    if len(py_tensor_1.shape) == 2:
        tmp_1 = py_tensor_1.numpy()
        tmp_2 = py_tensor_2.numpy()
    else:
        tmp_1 = cv2.cvtColor(py_tensor_1.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
        tmp_2 = cv2.cvtColor(py_tensor_2.numpy().transpose(1, 2, 0), cv2.COLOR_BGR2RGB)
    tmp = cv2.addWeighted(tmp_1, alpha_1, tmp_2, alpha_2, 0.0)
    cv2.imshow(window_str, tmp)
    cv2.waitKey(window_wait)

def visualize_pytorch_tensor(py_tensor: torch.Tensor, window_str: str="", window_wait: int=1) -> None:
    r""" Visualizes a PyTorch tensor with opencv. """
    if len(py_tensor.shape) == 2:
        tmp = py_tensor.numpy()
    else: 
        tmp = py_tensor.numpy().transpose(1, 2, 0)
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
    cv2.imshow(window_str, tmp)
    cv2.waitKey(window_wait)

def visualize_pil_image(pil_img: Image, window_str: str="", window_wait: int=1) -> None:
    r""" Visualizes a PIL image with opencv. """
    tmp = np.array(pil_img)[:, :, ::-1].copy() 
    cv2.imshow(window_str, tmp)
    cv2.waitKey(window_wait)
