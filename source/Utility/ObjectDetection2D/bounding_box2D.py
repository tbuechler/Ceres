import cv2
import torch
import numpy as np
from source.Utility.color import num2bgr


class BBox2D(object):
    """
    Wrapper for 2D bounding boxes.

    Attributes
    ----------
    x1 : float
        x coordinate of upper left point.
    y1 : float
        y coordinate of upper left point.
    x2 : float
        x coordinate of bottom right point.
    y2 : float
        y coordinate of bottom right point.
    category : int
        Category of the bounding box.
    score : float
        Computed classification score of bounding box.
    theta : float
        Rotation degree of bounding box.
    """
    def __init__(self,
        x1          :   float,
        y1          :   float,
        x2          :   float,
        y2          :   float,
        category    :   int=None,
        score       :   float=None,
        theta       :   float=None 
    ) -> None:
        self.x1         = x1
        self.y1         = y1
        self.x2         = x2
        self.y2         = y2
        self.category = int(category.item()) if isinstance(category, torch.Tensor) else category
        if category is not None and score is None:
            self.score      = 1.0
        else:
            self.score      = score
        self.theta      = theta
    
    def Width(self):
        """ Returns width of bounding box. """
        return abs(self.x2 - self.x1)
    
    def Height(self):
        """ Returns height of bounding box. """
        return abs(self.y2 - self.y1)

    def Points(self, down_cast=False):
        """ Returns both points of bounding box. If down_cast is true, then values are casted to integers. """
        if down_cast:
            return (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2))
        else:
            return (self.x1, self.y1), (self.x2, self.y2)


def visualize(img, boxes, objectList: str=None, imshow: bool=False, frame_name: str="", waitKey: int=1):
    """
    Visualization of multiple bounding boxes in a image along with their labels and scores.

    Arguments
    ---------
    img : array
        Input image the bounding boxes are drawing into.
    boxes : list[BBox2D] | BBox2D
        Can be either a BBox2D element only or a list.
    objectList : list[str]
        List of object names. Length of list should match category of all boxes.

    Returns
    -------
    img : array
        Image with bounding boxes drawn into. 
    """
    def visu_single_box( img, box: BBox2D, label: str=None, color=None ):
        line_thickness = int(round(0.001 * max(img.shape[0:2])))
        point1, point2 = box.Points(down_cast=True)

        ## Draw rectangle
        cv2.rectangle(img, point1, point2, color, line_thickness)
        if label is not None and isinstance(label, str):
            font_thickness  = max(line_thickness - 2, 1)
            s_size          = cv2.getTextSize(str('{:.0%}'.format(box.score)), 0, fontScale=float(line_thickness) / 3, thickness=font_thickness)[0]
            t_size          = cv2.getTextSize(label, 0, fontScale=float(line_thickness) / 3, thickness=font_thickness)[0]
            point2          = point1[0] + t_size[0] + s_size[0] + 15, point1[1] - t_size[1] - 3

            cv2.rectangle(img, point1, point2, color, -1)  # filled
            cv2.putText(img, '{}: {:.0%}'.format(label, box.score), (point1[0], point1[1] - 2), 0, float(line_thickness) / 3, [0, 0, 0],
                        thickness=font_thickness, lineType=cv2.FONT_HERSHEY_SIMPLEX)
        
    if isinstance(img, torch.Tensor):
        img = np.ascontiguousarray(img.numpy().transpose(1, 2, 0) * 255, dtype=np.uint8)

    bbs = [boxes] if isinstance(boxes, BBox2D) else boxes
    for box in bbs:
        if box.category is None:
            visu_single_box(img=img, box=box)
        elif objectList is None:
            visu_single_box(img=img, box=box, label=str(box.category), color=num2bgr(box.category))
        else:
            _label = str(box.category) if box.category >= len(objectList) else objectList[box.category] 
            visu_single_box(img=img, box=box, label=_label, color=num2bgr(box.category))
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if imshow:
        cv2.imshow(frame_name, img)
        cv2.waitKey(waitKey)
        
    return img
