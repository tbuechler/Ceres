import torch


class LearningRateScheduler:
    r"""
    # Learning Rate Scheduler

    Base class for all learning rate scheduler.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, lr: float):
        r"""
        Args:

        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `lr (float)`: 
            * Initial learning rate.
        """
        self.optimizer = optimizer
        self._lr       = lr

    def step(self, **kwargs):
        r"""
        One step forward in the learning rate schedule. This is required to be implemented by each learning rate scheduler.
        """
        raise NotImplementedError

    def get_lr(self):
        r""" Returns the learning rate the optimizer is using right now. """
        for g in self.optimizer.param_groups:
            return g['lr']

    def update_optimizer(self):
        r""" Set the learning rate of the optimizer. """
        for g in self.optimizer.param_groups:
            g['lr'] = self._lr

    @property
    def lr(self):
        return self.get_lr()

    @lr.setter
    def lr(self, value):
        r""" 
        If the learning rate is new assigned, the optimizer is updated simultaneously.
        """
        self._lr = value
        self.update_optimizer()


class WarmupLRScheduler(LearningRateScheduler):
    r"""
    # Warmup Learning Rate Scheduler

    Base class for warmup scheduler. Can be attached before every other policy. Currently the learning rate is increased linearly only.
    """
    def __init__(self, 
        optimizer       : torch.optim.Optimizer, 
        init_lr         : float, 
        peak_lr         : float, 
        warmup_steps    : int) -> None:
        r"""
        Args:
        
        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `peak_lr (float)`: 
            * Peak learning rate where policy should have ended.
        * `warmup_steps (int)`: 
            * Steps used from init_lr and peak_lr.
        """
        super().__init__(optimizer, init_lr)
        assert warmup_steps > 0, "Warmup steps must be greater than zero."

        self.update_steps   = 1
        self.lr             = init_lr
        self.init_lr        = init_lr
        self.peak_lr        = peak_lr
        self.warmup_steps   = warmup_steps
        self.warmup_rate    = (self.peak_lr - self.init_lr) / self.warmup_steps 

    def step(self):
        r""" One step forward in the learning rate scheduler. """
        if self.update_steps < self.warmup_steps:
            self.lr = self.init_lr + self.warmup_rate * self.update_steps
            self.update_steps += 1
        return self.lr

    def is_done(self):
        r""" Returns true if warmup schedule is over. """
        return self.update_steps >= self.warmup_steps
