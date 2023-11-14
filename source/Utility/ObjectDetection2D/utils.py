import torch
import torchvision
import torch.nn as nn

from source.Utility.ObjectDetection2D.bounding_box2D import BBox2D


class ClipBoxes(nn.Module):
    r"""
    Class to clip tensor bounding boxes to image resolution.

    Note that this function must be declared as a class where the clamping happens in the forward pass, since using torch.clamp individually will result in a high memory consumption. 

    The reason might be that there is no inplace operation available for torch.clamp on the backward path (see https://github.com/pytorch/pytorch/issues/33373#issuecomment-587638395). 
    """
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        r"""
        Clips input bounding boxers to image resolution.
        
        Args:
            boxes (torch.Tensor): Tensor of shape [B, num_boxes, 4] in ''xyxy''
                format. 
            img (torch.Tensor): Tensor of shape [B, C, H, W].
        """
        _, _, height, width = img.shape
        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width - 1)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height - 1)
        return boxes
        

def tensor_to_BBox2D(
        decoded_output  :   torch.Tensor, # shape [batch, num_boxes, [x1, y1, x2, y2]]
        classification  :   torch.Tensor, # shape [batch, num_boxes, [class1_score, class2_score, ...]]
        score_threshold :   float = 0.0,
        iou_threshold   :   float = 0.0
    ):
    """
    Converts tensors to BBox2D objects and filters out boxes with a low classification score (below score_threshold).
    Additionally, NMS (Non-Maximum suppresion) is performed.

    Arguments
    ---------
    decoded_output : torch.Tensor
        Decoded output from a network that represents multiple bounding boxes with the shape of [batch, num_boxes, [x1, y1, x2, y2]].
    classification : torch.Tensor
        Classification result of all bounding boxes with the shape of [batch, num_boxes, [class1_score, class2_score, ...]].
    score_threshold : float
        Threshold under which all bounding boxes will be filtered out.
    iou_threshold : float
        IoU (intersection of union) threshold is used by NMS.
    """
    ## Filter using classification score and threshold
    individual_scores = torch.max(classification, dim=2, keepdim=True)[0] # Dimension 2, because of shape of (B, num_anchors, num_classes)
    filtered_scores   = (individual_scores > score_threshold)[:, :, 0]

    # Return element containing bounding box information, i.e. coords, classification, certainty, ...
    to_return = []

    # Iterate over all batches
    for i_batch in range(classification.shape[0]):
        to_return.append([])

        ## Get all classifications from boxes which exceed threshold and fold to shape (class_scores, num_boxes)
        classification_per_box = classification[i_batch, filtered_scores[i_batch, :], ...].permute(1, 0)
        transformed_boxes      = decoded_output[i_batch, filtered_scores[i_batch, :], ...]
        class_scores_per_box   = individual_scores[i_batch, filtered_scores[i_batch, :], ...]
        ## Get max score and classification id from selected boxes
        _score, _class_id = classification_per_box.max(dim=0)
        ## Apply Non-Maximum Suppression
        nms_idxs = torchvision.ops.boxes.batched_nms(
            boxes=transformed_boxes,
            scores=class_scores_per_box[:, 0],
            idxs=_class_id,
            iou_threshold=iou_threshold
        )

        to_return[i_batch] = [
            BBox2D(
                x1          =   float(transformed_boxes[nms_idx, :][0]),
                y1          =   float(transformed_boxes[nms_idx, :][1]),
                x2          =   float(transformed_boxes[nms_idx, :][2]),
                y2          =   float(transformed_boxes[nms_idx, :][3]),
                category    =   int(_class_id[nms_idx]),
                score       =   float(_score[nms_idx])
            )
            for nms_idx in nms_idxs
        ]

    return to_return
