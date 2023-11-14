r"""
Implementation of 

* **L1-norm** also known as least absolute deviations/errors (LAD/LAE)
* **L2-norm** also known as least squares error (LSE)
* **SmoothL1** which can be interpreted as a combination of the L1-norm and L2-norm

![L1L2_Overview](../../assets/loss/l1l2/overview.png)

"""
import torch


class L1_Loss:
    r"""
    # L1 loss function

    L1 loss function defined as:
    $$
    f(a,b) = | a - b |.
    $$
    """
    def __init__(self, reduction: str=None) -> None:
        r"""
        Args:

        * `reduction (str)`:
            * reduction: 'none' | 'mean' | 'sum'
                * 'none': No reduction will be applied to the output.
                * 'mean': The output will be averaged.
                * 'sum': The output will be summed.
        """
        self.reduction = reduction

    def __call__(self, 
        prediction  :   torch.Tensor,
        target      :   torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:

        * `prediction (torch.Tensor)`:
            * Input tensor of any shape
        * `target (torch.Tensor)`:
            * Target value tensor with the same shape as input.

        Returns:
            
        * The loss with the reduction option applied.
        """
        ## Computation of L1 loss
        loss = torch.abs(torch.sub(prediction, target))        
        ## Reduction handling
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
        

class SmoothL1_Loss:
    r"""
    # Smooth L1 loss function
    
    Smooth L1 loss defined as:
    $$
    f(a,b) = 
     \begin{cases}
       0.5 \cdot x^2 / \beta   &\quad\text{if}\quad \text{abs}(x) < \beta\\
       \text{abs}(x) - 0.5 \cdot \beta   &\quad\text{otherwise,}\\
     \end{cases}
     $$
    SmoothL1 loss can be seen as exactly L1 loss, but with the $\text{abs}(x) < \beta$ portion replaced with a quadratic function such that at $\text{abs}(x) = \beta$, its slope is 1. The quadratic segment smooths the L1 loss near $x = 0$.
    """
    def __init__(self, beta: float, reduction: str=None) -> None:
        r"""
        Args:

        * beta (float):
            * L1 to L2 change point. For beta values < 1e-5, L1 loss is computed.
        * reduction (str):
            * reduction: 'none' | 'mean' | 'sum'
                * 'none': No reduction will be applied to the output.
                * 'mean': The output will be averaged.
                * 'sum': The output will be summed.
        """
        self.beta       = beta    
        self.reduction  = reduction

    def __call__(self, 
        prediction  :   torch.Tensor,
        target      :   torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:

        * `prediction (torch.Tensor)`:
            * Input tensor of any shape
        * `target (torch.Tensor)`:
            * Target value tensor with the same shape as input.

        Returns:
            
        * The loss with the reduction option applied.
        """

        # If $\beta = 0$, then `torch.where(...)` will result in `NaN` gradients when the chain rule is applied due to pytorch implementation details. To avoid this issue, we define small values of beta to be exactly L1 loss.
        if self.beta < 1e-5:
            loss = torch.abs(prediction - target)
        else:
            n    = torch.abs(prediction - target)
            cond = n < self.beta
            loss = torch.where(cond, 0.5 * n ** 2 / self.beta, n - 0.5 * self.beta)
        ## Reduction handling
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class L2_Loss:
    r"""
    # L2 loss function

    L2 loss function defined as:
    $$
    f(a,b) = ( a - b )^2.
    $$
    """
    def __init__(self, reduction: str=None) -> None:
        r"""
        Args:

        * `reduction (str)`:
            * reduction: 'none' | 'mean' | 'sum'
                * 'none': No reduction will be applied to the output.
                * 'mean': The output will be averaged.
                * 'sum': The output will be summed.
        """
        self.reduction = reduction

    def __call__(self, 
        prediction  :   torch.Tensor,
        target      :   torch.Tensor
    ) -> torch.Tensor:
        r"""
        Args:

        * `prediction (torch.Tensor)`:
            * Input tensor of any shape
        * `target (torch.Tensor)`:
            * Target value tensor with the same shape as input.

        Returns:
            
        * The loss with the reduction option applied.
        """
        ## Computation of loss
        loss = torch.pow(torch.sub(prediction, target), 2)
        ## Reduction handling
        if self.reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
