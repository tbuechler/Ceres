import torch

class AverageMeter(object):
    r""" Computes and stores the average of a specific type of value. """
    def __init__(self, name):
        self.name = name
        self.reset()

    def reset(self):
        self.val = torch.zeros(1)
        self.avg = torch.zeros(1)
        self.sum = torch.zeros(1)
        self.count = 0

    def update(self, val):
        self.val = val.detach()
        if torch.isnan(val):
            return
        self.sum += self.val
        self.count += 1
        self.avg = self.sum / self.count

    def __call__(self):
        return self.avg

    def __str__(self):
        return "{} {:.5f} (Avg: {:.5f}, n: {})".format(
            self.name,
            self.val.item(),
            self.avg.item(),
            self.count
        )
