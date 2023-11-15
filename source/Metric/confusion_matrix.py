import torch
import torch.nn as nn


class ConfusionMatrix(nn.Module):
    r"""
    Class that creates a confusion matrix internally.
    """
    def __init__(self, num_classes: int, ignore_val: int = None):
        r"""
        Args:
        
        * `num_classes (int)`: 
            * Number of different classes to be considered.
        * `ignore_val (int)`: 
            * Unique value that makes it possible to ignore some values.
        """
        super(ConfusionMatrix, self).__init__()
        self.num_classes: int = num_classes
        self.ignore_val: int  = ignore_val
        self.current_matrix: torch.Tensor = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.current_groundtruth: torch.Tensor = None
        self.current_prediction: torch.Tensor = None

    def get_matrix(self) -> torch.Tensor:
        r""" Returns the confusion matrix represented as PyTorch Tensor. """
        return self.current_matrix

    def get_precisions(self) -> torch.Tensor:
        r"""Computes the precision out of the matrix and returns it. """
        true_positives = self.current_matrix.diag()
        precisions = true_positives / self.current_matrix.sum(axis=0) # TP / TP + FP
        return precisions

    def get_recalls(self) -> torch.Tensor:
        r""" Computes the recall out of the matrix and returns it. """
        true_positives = self.current_matrix.diag()
        recalls = true_positives /  self.current_matrix.sum(axis=1) # TP / TP + FN
        return recalls

    def _set_and_filter_params(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> None:
        r""" 
        Adapts the prediction and groundtruth input for further computation. Additionally, the
        input is adapted according to the value of `ignore_value`.
        """
        self.current_prediction = torch.argmax(prediction, dim=1).flatten()
        self.current_groundtruth = groundtruth.flatten()
        
        if self.ignore_val is not None:
             indices_to_keep = [self.current_groundtruth != self.ignore_val]
             self.current_groundtruth = self.current_groundtruth[indices_to_keep]
             self.current_prediction = self.current_prediction[indices_to_keep]

    def update_current_matrix(self) -> None:
        r""" 
        Updates the matrix according to the last prediction and groundtruth data internally.
        """
        classes_stacked = torch.stack((self.current_groundtruth, self.current_prediction), dim=1)
        classes_combinations, combinations_counts  = torch.unique(classes_stacked, dim=0, return_counts=True)

        for combination, count in zip(classes_combinations, combinations_counts):
            true_class, predicted_class = combination.tolist()
            self.current_matrix[true_class, predicted_class] += count

    def forward(self, prediction: torch.Tensor, groundtruth: torch.Tensor) -> None:
        r""" Updates the confusion matrix according to the prediction. """
        self._set_and_filter_params(prediction, groundtruth) 
        self.update_current_matrix()