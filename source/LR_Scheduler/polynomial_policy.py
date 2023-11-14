r"""
Implementation of the polynomial decay learning rate scheduler.

![PolyLR_Overview](../../assets/lr_scheduler/PolyLRScheduler.png)
"""
import torch
import source.LR_Scheduler.lr_scheduler_base as LRScheduler_Base


class PolyLRScheduler(LRScheduler_Base.LearningRateScheduler):
    r"""
    # Polynomial decay schedule.
    
    Decays the learning rate of each parameter group by 
    
    $$
    \gamma = (1. - \frac{u}{n})^p
    $$

    with $u: \text{update_steps}$, $n: \text{max_iter}$ and $p: \text{poly}$ at every step_size epochs. When last_epoch=-1, sets initial lr as lr.
    """
    def __init__(self, 
            optimizer    : torch.optim.Optimizer,
            init_lr      : float,
            max_iter     : int, 
            poly         : float,
            peak_lr      : float = 1e-2,
            warmup_steps : int = 0,
            do_warmup    : bool = False
    ) -> None:
        r"""
        Args:
        
        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `max_iter (int)`: 
            * Max. number of iteration.
        * `poly (float)`: 
            * Power scale how hard the learning rate is decreased.
        * `do_warmup (bool)`: 
            * If true, warmup policy is used in the beginning.
        * `peak_lr (float)`: 
            * Used for warmup policy which is equal to the initial rate of OnPlateauLRScheduler then.
        * `warmup_step (int)`: 
            * Number of steps for the warmup policy.
        """
        super().__init__(optimizer, init_lr)
        if do_warmup:
            self.warmup = LRScheduler_Base.WarmupLRScheduler(
                optimizer=optimizer,
                init_lr=init_lr,
                peak_lr=peak_lr,
                warmup_steps=warmup_steps
            )
        else:
            self.warmup = None

        self.init_lr        = peak_lr if do_warmup else init_lr
        self.max_iter       = max_iter
        self.poly           = poly
        self.update_steps   = 0
        
    def step(self):
        r""" One step forward in the learning rate schedule. """
        if self.warmup is not None and not self.warmup.is_done():
            self.warmup.step()
        else:
            ## 
            self.lr = self.init_lr * (1. - (self.update_steps / self.max_iter) ) ** self.poly
            
        self.update_steps += 1
