r"""
Implementation of the learning rate scheduler proposed in: [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf).

![TransformerLR_Overview](../../assets/lr_scheduler/TransformerLRScheduler.png)
"""
import math
import torch
import source.LR_Scheduler.lr_scheduler_base as LRScheduler_Base


class TransformerLRScheduler(LRScheduler_Base.LearningRateScheduler):
    r"""
    # Transformer LR Scheduler

    The learning rate is varied over the training according to the formula:

    $$
    lr = d_{model}^{-0.5} \cdot min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{1-5}).
    $$

    Thus, the learning rate is increased linearly for the first $warmup\_steps$ training steps, and
    decreased thereafter proportionally to the inverse square root of the $step\_num$.
    """
    def __init__(self,
        optimizer       : torch.optim.Optimizer,
        init_lr         : float,
        peak_lr         : float,
        final_lr        : float,
        final_lr_scale  : float,
        warmup_steps    : int,
        decay_steps     : int) -> None:
        r"""
        Args:

        * `optimizer (torch.optim.Optimizer)`: 
            * Wrapped optimizer.
        * `init_lr (float)`: 
            * Initial learning rate.
        * `peak_lr (float)`: 
            * Maximum learning rate for the warmup phase.
        * `final_lr (float)`: 
            * Final learning rate.
        * `final_lr_scale (float)`: 
            * Scale factor for the final learning rate.
        * `warmup_steps (int)`: 
            * Number of steps for the warmup phase.
        * `hold_steps (int)`: 
            * Number of steps the learning rate stays equal to peak_lr.
        * `decay_steps (int)`: 
            * Number of steps the learning rate is decayed linearly.
        """

        super(TransformerLRScheduler, self).__init__(optimizer, init_lr)
        self.final_lr = final_lr
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps

        self.warmup_rate = self.peak_lr / self.warmup_steps
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = init_lr
        self.update_steps = 0

    def _decide_stage(self):
        r""" 
        Determines the current state according to the current step. 
        
        * State 1: Increase learning rate according to warmup rate.
        * State 2: Decrease learning rate proportionally to the inverse square root of the step number.
        * State 3: Keep learning rate constant.
        """
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        if self.warmup_steps <= self.update_steps < self.warmup_steps + self.decay_steps:
            return 1, self.update_steps - self.warmup_steps

        return 2, None

    def step(self):
        r""" One step forward in the learning rate schedule. """
        self.update_steps += 1
        stage, steps_in_stage = self._decide_stage()
        if stage == 0:
            self.lr = self.update_steps * self.warmup_rate
        elif stage == 1:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 2:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")
        
        return self.lr