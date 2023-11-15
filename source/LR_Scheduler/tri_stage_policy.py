r"""
Implementation of the learning rate scheduler proposed in: [SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition](https://arxiv.org/pdf/1904.08779.pdf).

![TriStageLR_Overview](../../assets/lr_scheduler/TriStageLRScheduler.png)
"""
import math
import torch
import source.LR_Scheduler.lr_scheduler_base as LRScheduler_Base


class TriStageLRScheduler(LRScheduler_Base.LearningRateScheduler):
    r"""
    # Tristage learning rate scheduler
    
    A learning rate schedule policy that first warms up, then holds the value constant and 
    finally exponentially decay the learning rate until it reaches some final value.
    """
    def __init__(self,
        optimizer       : torch.optim.Optimizer,
        init_lr         : float,
        peak_lr         : float,
        final_lr        : float,
        init_lr_scale   : float,
        final_lr_scale  : float,
        warmup_steps    : int,
        hold_steps      : int,
        decay_steps     : int) -> None:
        r"""
        Args:
        
        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `peak_lr (float)`: 
            * Maximum learning rate for the warmup phase.
        * `final_lr_scale (float)`: 
            * Scale factor for the final learning rate.
        * `warmup_steps (int)`: 
            * Number of steps for the warmup phase.
        * `hold_steps (int)`: 
            * Number of steps the learning rate stays equal to peak_lr.
        * `decay_steps (int)`: 
            * Number of steps the learning rate is decayed linearly.
        """
        super().__init__(optimizer, init_lr)
        self.init_lr = init_lr
        self.init_lr *= init_lr_scale
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.update_steps = 0

    def _decide_stage(self):
        r""" 
        Determine current stage by the current step. 
        
        * State 1: Learning rate is increased by warmup rate.
        * State 2: Learning rate is kept constant equal to the peak value.
        * State 3: Learning rate is decreased exponentially.
        * State 4: Learning rate is kept constant.
        """
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps
        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps
        if self.update_steps <= offset + self.decay_steps:
            return 2, self.update_steps - offset

        offset += self.decay_steps
        return 3, self.update_steps - offset

    def step(self):
        r""" One step forward in the learning rate schedule. """
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.update_steps += 1
