r"""
Implementation of the one cycle learning rate scheduler.

![OneCycleScheduler_Overview](../../assets/lr_scheduler/OneCycleScheduler.png)
"""
import torch
import source.LR_Scheduler.lr_scheduler_base as LRScheduler_Base


class OneCycleLR(LRScheduler_Base.LearningRateScheduler):
    r"""    
    # One Cycle Learning Rate Policy

    """
    def __init__( self,
            optimizer       : torch.optim.Optimizer,
            init_lr         : float,
            max_lr          : float,
            steps_per_epoch : int,
            epochs          : float
    ) -> None:
        r"""
        Args:
        
        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `max_lr (float)`: 
            * Maximal learning rate.
        * `steps_per_epoch (int)`: 
            * Number of steps within one epoch.
        * `epochs (float)`: 
            * Number of epoch in this session.
        """
        super(OneCycleLR, self).__init__(optimizer, init_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs
        )

    def step(self):
        r""" One step forward in the learning rate schedule. """
        self.lr_scheduler.step()
