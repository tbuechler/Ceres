import torch
import torch.nn as nn

def calculate_IoU_anchors(anchors: torch.Tensor, gt: torch.Tensor):
    """
    Computes the Intersection-of-Union between all anchor boxes and the given ground truth data.

    Parameters
    ----------
    anchors : torch.Tensor
        Precomputed anchor boxes. Shape: (num_anchors, 4), i.e. (x1, y1, x2, y2).
    gt : torch.Tensor
        Ground truth data with available bounding boxes in the image. Shape: (num_boxes, 4), i.e. (x1, y1, x2, y2).

    Returns
    -------
    IoU : torch.Tensor
        Computed IoU between each anchor box and each box in the ground truth data.
        Thus the output shape must be: (num_anchors, num_bb_gt). 
        i.e. [(49104, 4), (20, 4)] -> (49104, 20)   
    """
    ## Compute width and height of the intersection part
    ## Width: min(x2, x2') - max(x1, x1')
    inter_width = torch.min(torch.unsqueeze(anchors[:, 2], dim=1), gt[:, 2]) - torch.max(torch.unsqueeze(anchors[:, 0], dim=1), gt[:, 0])
    inter_width = torch.clamp(inter_width, min=0) # Just in case of anchor boxes outside of image size
    ## Height: min(y2, y2') - max(y1, y1')
    inter_height = torch.min(torch.unsqueeze(anchors[:, 3], dim=1), gt[:, 3]) - torch.max(torch.unsqueeze(anchors[:, 1], dim=1), gt[:, 1])
    inter_height = torch.clamp(inter_height, min=0) # Just in case of anchor boxes outside of image size
    ## Area of intersection: Height*Width
    inter_area = inter_height * inter_width

    ## Compute area of union part
    ## Area of anchor box + Area of GT box - intersection
    anchor_area = torch.unsqueeze(((anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])), dim=1) # (x2-x1)[Width] * (y2-y1)[Height]
    bb_area     = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) # (x2-x1)[Width] * (y2-y1)[Height]
    union_area  = anchor_area + bb_area - inter_area
    
    return (inter_area / union_area) 

class Focal_SmoothL1_Loss_OD(nn.Module):
    def __init__(self, alpha: float=0.25, gamma: float=2.0, device: str='cpu') -> None:
        super().__init__()
        self.alpha   = alpha
        self.gamma   = gamma
        self.device = device

    def forward(self, regression: torch.Tensor, classification: torch.Tensor, anchors: torch.Tensor, gt_BB: torch.Tensor):
        """
        Computes the focal loss for the classification path and the SmoothL1 loss for the regression path.

        Parameters
        ----------
        regression : torch.Tensor
            Prediction of location of objects. Shape: [B, n_anchors, 4], i.e. (x1, y1, x2, y2).
        classification : torch.Tensor
            Prediction of the class of each object. Shape: [B, n_anchors, num_classes].
        anchors : torch.Tensor
            Tensor containing all precomputed anchor boxes for all scales. Shape: [1, n_anchors, 4], i.e. (x1, y1, x2, y2).
        gt_BB : torch.Tensor
            Ground truth data for each image. Shape: [B, max_bbs_in_image_batch, 5], i.e. (x1, y1, x2, y2, c).
        
        Returns
        -------
        class_loss
            Classification loss using FocalLoss.
        regression_loss
            Regression loss using SmoothL1-Loss.
        """
        ## Loss container for regression and classification part
        classification_loss = []
        regression_loss     = []

        ## Transform anchors from point coordinate structure to center height/width structure
        ## (x1, y1, x2, y2) -> (x_c, y_c, width, height)
        _anchors = torch.zeros(anchors[0, :, :].shape).to(torch.device(self.device))
        _anchors[:, 3] = abs(anchors[0, :, 0] - anchors[0, :, 2]) # Width  = abs(x1 - x2)
        _anchors[:, 2] = abs(anchors[0, :, 1] - anchors[0, :, 3]) # Height = abs(y1 - y2)
        _anchors[:, 1] = anchors[0, :, 1] + (0.5 * _anchors[:, 2]) # y_c = y1 + 0.5*height
        _anchors[:, 0] = anchors[0, :, 0] + (0.5 * _anchors[:, 3]) # x_c = x1 + 0.5*width

        ## Iterate over each element in the current batch
        for i_batch in range(regression.shape[0]):
            
            ## Fetch information for current element
            b_classification = torch.clamp(input = classification[i_batch, :, :], min = 1e-4 ,max = 1 - 1e-4)
            b_regression     = regression[i_batch, :, :]
            # Filter out invalid GT data, indicated by -1 value
            b_annotation = gt_BB[i_batch, :, :]
            b_annotation = b_annotation[b_annotation[:, 4] != -1]

            ###
            ## Case that no ground truth bounding box is available for this element.
            ## This is equal to the else part of the general formula of focal loss.
            ## FL(p) = -(1-alpha) * p^gamma * log(1-p) if y != 1
            ###
            if b_annotation.shape[0] == 0:
                alpha_factor = torch.ones_like(b_classification) * (1. - self.alpha)
                focal_weight = alpha_factor * torch.pow(b_classification, self.gamma)

                # Cross Entropy part
                bce = -(torch.clamp(torch.log(1.0 - b_classification), min=-100))

                # Class loss
                class_loss = focal_weight * bce

                # append to global lists
                classification_loss.append(class_loss.sum())
                regression_loss.append(torch.tensor(0).to(torch.float32))  # Regression does not matter thats why the loss is zero

                # Nothing to do else, skip to next element
                continue


            ###
            ## Case there are bounding boxes given in the ground truth data.
            ###
            ## 1. Determine target anchors according to their IoU with GT bb
            IoU_anchors_bbs = calculate_IoU_anchors(anchors=anchors[0, :, :], gt=b_annotation[:, :4])
            # Max IoU and index for each anchor box
            IoU_max_val, IoU_max_idx = torch.max(IoU_anchors_bbs, dim=1) # (num_anchors)

            ###
            ## Classification loss according to FocalLoss formula
            ###
            # Compute target elements in anchors
            targets = torch.ones_like(b_classification, device=self.device) * -1    # (num_anchors, num_classes)

            # Filter out negative and positive samples in target vector
            # Negative: IoU < 0.4
            targets[torch.lt(IoU_max_val, 0.4), :] = 0
            # Positive: IoU > 0.5
            positive_indices    = torch.ge(IoU_max_val, 0.5)
            n_positive_samples  = positive_indices.sum()

            ##  Per each element in IoU_max_idx, the corresponding box in b_annotation is chosen according to the IoU value.
            ##  assigned_annotations will contain the bounding box with the highest IoU per anchor box
            ##  Thus, the shape of assigned_annotations will be (num_anchors, 5) [5: coords + class value]
            assigned_annotations = b_annotation[IoU_max_idx, :]

            ## Set every positive element in target to zero except the assigned annotations
            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            ## From here compute the actual focal loss 
            alpha_factor = torch.ones_like(targets, device=self.device) * self.alpha # Not 1 - alpha, because all elements with value one are positive, with zero are negative.

            # Assign each element with alpha when value is one, if zero assign with 1-alpha according to focal loss
            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            # Same goes for focal_weight
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - b_classification, b_classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

            # Cross entropy part
            # Note: Using only torch.log without clamp can lead to NaN values.
            #       See https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/717
            bce = -(targets * torch.clamp(torch.log(b_classification), min=-100) + (1.0 - targets) * torch.clamp(torch.log(1.0 - b_classification), min=-100))
            
            # Class loss
            class_loss = focal_weight * bce

            # Every entry which is not -1, which indicates an invalid box, is assigned with the class error, else zero.
            zeros = torch.zeros_like(class_loss, device=self.device)

            class_loss = torch.where(torch.ne(targets, -1.0), class_loss, zeros)

            # Append average loss across anchor boxes
            classification_loss.append(class_loss.sum() / torch.clamp(n_positive_samples, min=1.0))

            ###
            ## Regression loss
            ###
            if n_positive_samples > 0:
                # Compute regression loss only if positive entries are available
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = _anchors[positive_indices, 3]
                anchor_heights_pi = _anchors[positive_indices, 2]
                anchor_ctr_x_pi = _anchors[positive_indices, 0]
                anchor_ctr_y_pi = _anchors[positive_indices, 1]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # efficientdet style
                gt_widths = torch.clamp(gt_widths, min=1)
                gt_heights = torch.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = torch.log(gt_widths / anchor_widths_pi)
                targets_dh = torch.log(gt_heights / anchor_heights_pi)
                
                targets = torch.stack((targets_dx, targets_dy, targets_dh, targets_dw))
                targets = targets.t()

                regression_diff = torch.abs(targets - b_regression[positive_indices, :])
                
                # Unclear why this was written in this way.
                # It is basically equal to the SmoothL1 loss with beta = 1./9.
                _regression_loss = torch.where(
                    torch.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_loss.append(_regression_loss.mean())
            else:
                regression_loss.append(
                    torch.tensor(0, device=self.device).to(torch.float32)
                )


        # If regression loss is not multiplicated with a certain value it will vanish in the total loss
        regression_loss_multiplicator = 50.0
        return  torch.stack(classification_loss).mean(dim=0, keepdim=True), \
                torch.stack(regression_loss).mean(dim=0, keepdim=True) * regression_loss_multiplicator
