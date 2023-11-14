import torch


def abs_error(prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    r"""
    Absolute error between an approximated value and the exact value.

    Args:
    
    * `prediction (torch.Tensor)`: 
        * Approximated values represented as an tensor.
    * `groundtruth (torch.Tensor)`: 
        * Exact value which is also presented as an tensor and has the exact same shape as the prediction.
    """
    return torch.mean(torch.abs(groundtruth - prediction), groundtruth)
   
def abs_rel_error(prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    r"""
    Realtive error between an approximated value and the exact value.

    Args:
    
    * `prediction (torch.Tensor)`: 
        * Approximated values represented as an tensor.
    * `groundtruth (torch.Tensor)`: 
        * Exact value which is also presented as an tensor and has the exact same shape as the prediction.
    """
    return torch.mean(torch.div(torch.abs(groundtruth - prediction)), groundtruth)

def sq_rel_error(prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    r"""
    Squared relative error between an approximated value and the exact value.

    Args:
    
    * `prediction (torch.Tensor)`: 
        * Approximated values represented as an tensor.
    * `groundtruth (torch.Tensor)`: 
        * Exact value which is also presented as an tensor and has the exact same shape as the prediction.
    """
    return torch.mean(torch.div((groundtruth - prediction) ** 2), groundtruth)

def rmse(prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    r"""
    Computation of the root-mean-square error.

    Args:
    
    * `prediction (torch.Tensor)`: 
        * Approximated values represented as an tensor.
    * `groundtruth (torch.Tensor)`: 
        * Exact value which is also presented as an tensor and has the exact same shape as the prediction.
    """
    rmse = (groundtruth - prediction) ** 2
    return torch.sqrt(torch.mean(rmse))

def log_rmse(prediction: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    r"""
    Computation of the root-mean-squared-log error.

    Args:
    
    * `prediction (torch.Tensor)`: 
        * Approximated values represented as an tensor.
    * `groundtruth (torch.Tensor)`: 
        * Exact value which is also presented as an tensor and has the exact same shape as the prediction.
    """
    rmse = (torch.log(groundtruth) - torch.log(prediction)) ** 2
    return torch.sqrt(torch.mean(rmse))
