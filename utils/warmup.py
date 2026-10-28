"""
MIT License

Copyright (c) 2019 Ildoo Kim

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """Warm up to base_lr * multiplier, then continue with after_scheduler.

    A multiplier of 1 warms up from zero; larger multipliers start at base_lr.
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier must be at least 1.')
        if total_epoch < 1:
            raise ValueError('total_epoch must be positive.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                    return list(self.after_scheduler.base_lrs)
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        self.last_epoch = self.last_epoch + 1 if epoch is None else epoch
        if self.last_epoch <= self.total_epoch:
            warmup_lr = self.get_lr()
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if not self.finished:
                for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                    group['lr'] = base_lr * self.multiplier
                self.finished = True
            if epoch is None:
                self.after_scheduler.step(metrics)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

    def step(self, epoch=None, metrics=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)

    def state_dict(self):
        state = super().state_dict()
        state['after_scheduler'] = self.after_scheduler.state_dict() if self.after_scheduler else None
        return state

    def load_state_dict(self, state_dict):
        state = dict(state_dict)
        after = state.pop('after_scheduler', None)
        if self.after_scheduler is not None and after is not None:
            self.after_scheduler.load_state_dict(after if isinstance(after, dict) else after.state_dict())
        super().load_state_dict(state)
