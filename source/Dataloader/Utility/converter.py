import sys
import cv2
import torch
import numpy as np

from typing import Tuple


def labImg2Edge(label: np.array, edge_pad: bool=True, edge_size: int=4, thresholds: Tuple[float, float]=(0.1, 0.2)):
    r"""
    Method to compute edges in a image by using the Canny Edge Detector.

    Args:

    * `label (np.arry)`:
        * Input image as numpy array.
    * `edge_pad (bool)`:
        * If set to true, the edge is padded additionally using a predefined kernel size.
    * `edge_size (int)`:
        * Dilation kernel size.
    * `thresholds (Tuple[float, float])`:
        * Thresholds used by the Canny Edge Detector.
    """
    edge = cv2.Canny(label, thresholds[0], thresholds[1])
    kernel = np.ones((edge_size, edge_size), np.uint8)
    y_k_size = 6
    x_k_size = 6
    if edge_pad:
        edge = edge[y_k_size:-y_k_size, x_k_size:-x_k_size]
        edge = np.pad(edge, ((y_k_size,y_k_size),(x_k_size,x_k_size)), mode='constant')
    return (cv2.dilate(edge, kernel, iterations=1)>50)*1.0


def labImg2EdgeTorch(label: torch.tensor, edge_pad: bool=True, edge_size: int=4, thresholds: Tuple[float, float]=(0.1, 0.2)):
    r"""
    Method to compute edges in a image by using the Canny Edge Detector having
    a PyTorch as input.

    Args:

    * `label (torch.Tensor)`:
        * Input image as PyTorch Tensor.
    * `edge_pad (bool)`:
        * If set to true, the edge is padded additionally using a predefined kernel size.
    * `edge_size (int)`:
        * Dilation kernel size.
    * `thresholds (Tuple[float, float])`:
        * Thresholds used by the Canny Edge Detector.
    """
    return torch.from_numpy(labImg2Edge(label.cpu().detach().numpy().astype(np.uint8), edge_pad=edge_pad, edge_size=edge_size, thresholds=thresholds))


def polygon2BB(polygon: list) -> int:
    r"""
    Converts a list of points of a polygon into a 2D bounding box by selecting the border points.

    Args:

    * ´polygon (list)`:
        * List of polygon coordinates: `[(x1, y1), (x2, y2), ...]`.

    Returns:

    * Bounding box coordinates: `[x1, y1, x2, y2]`.
    """
    bb = [sys.maxsize, sys.maxsize, -sys.maxsize, -sys.maxsize]
    for poly in polygon:
        bb[0] = poly[0] if poly[0] < bb[0] else bb[0] 
        bb[1] = poly[1] if poly[1] < bb[1] else bb[1] 
        bb[2] = poly[0] if poly[0] > bb[2] else bb[2] 
        bb[3] = poly[1] if poly[1] > bb[3] else bb[3] 
    return bb