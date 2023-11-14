r"""
Implementation of the learning rate scheduler reduction on plateau.

![OnPlateauLR_Overview](../../assets/lr_scheduler/OnPlateauLRScheduler.png)
"""
import torch
from source.Logger.logger import log_warning
import source.LR_Scheduler.lr_scheduler_base as LRScheduler_Base


class OnPlateauLRScheduler(LRScheduler_Base.LearningRateScheduler):
    r"""    
    # Learning Rate Scheduler On Plateau

    Reduce learning rate when a metric has stopped improving. Models often benefit from reducing the learning rate by a factor of 2-10 once learning stagnates. This scheduler reads a metrics quantity and if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced (see PyTorch).

    This implementation of the OnPlataeuLRScheduler supports only one interpretation of the metric. An increase of the metric value at each step is interpreted as a degradation of the metric.
    """
    def __init__( self,
            optimizer    : torch.optim.Optimizer,
            init_lr      : float,
            patience     : int,
            factor       : float,
            peak_lr      : float = 1e-2,
            warmup_steps : int = 0,
            eps          : float = 1e-4,
            do_warmup    : bool = False
    ) -> None:
        r"""
        Args:
        
        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `patience (int)`: 
            * Recalculate learning rate if no improvement for `patience` steps is noted.
        * `factor (float)`: 
            * Factor to reduce the learning rate.
        * `eps (float)`: 
            * Improvement if new value is better than old value including epsilon.
        * `do_warmup (bool)`: 
            * If true, warmup policy is used in the beginning.
        * `peak_lr (float)`: 
            * Used for warmup policy which is equal to the initial rate of OnPlateauLRScheduler then.
        * `warmup_step (int)`: 
            * Number of steps for the warmup policy.
        """
        super(OnPlateauLRScheduler, self).__init__(optimizer, init_lr)

        if do_warmup:
            self.warmup = LRScheduler_Base.WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps
            )
        else:
            self.warmup = None
            self.lr    =  init_lr
        
        self.patience   = patience
        self.factor     = factor
        self.eps        = eps

        self.first_step = 0

        self.val_loss = 100.0
        self.count = 0

    def step(self, val_loss: float):
        r""" One step forward in the learning rate schedule. """
        if val_loss is not None:
            ## If the warmup policy was setup and is not done yet, 
            ## perform this policy instead of the OnPlateau.
            if self.warmup is not None and not self.warmup.is_done():
                self.warmup.step()

            else:
                ## The first step of the OnPlateau policy is handled 
                ## separately because no improvement or degradation 
                ## of the metric is noticeable.
                if self.first_step == 0:
                    self.val_loss = val_loss
                    self.first_step += 1

                else:
                    ## For later steps the counter of the LR Scheduler is 
                    ## updated according to an improvement or degradation 
                    ## of the metric.            
                    if self.val_loss < (val_loss + self.eps):
                        self.count += 1
                    else:
                        self.count = 0
                        self.val_loss = val_loss

                    ## The learning rate is updated using the factor if the
                    ## number of patience steps is reached.
                    if self.patience == self.count:
                        self.count = 0
                        self.lr *= self.factor
        else:
            log_warning("NaN validation loss.", show_stack=True)
        return self.lr