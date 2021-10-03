from torch.optim.lr_scheduler import _LRScheduler, StepLR

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, warmUp=False, power=0.9, last_epoch=-1, min_lr=1e-5):
        self.power = power
        self.warmup = warmUp
        self.max_iters = max_iters  # avoid zero lr
        self.min_lr = min_lr
        self.warmup_iters = int(0.1 * self.max_iters) if warmUp else 0
        super(PolyLR, self).__init__(optimizer, last_epoch)   
    
    def get_lr(self):
        if self.warmup and self.last_epoch <= self.warmup_iters:
            return [ max( base_lr * ( self.last_epoch / self.warmup_iters ), self.min_lr ) for base_lr in self.base_lrs ]
        else:
            return [ max( base_lr * ( max(1 - (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters), 0) )**self.power, self.min_lr) for base_lr in self.base_lrs ]
    
    def show_lr(self):
        if self.warmup and self.last_epoch <= self.warmup_iters:
            return [ max( base_lr * ( self.last_epoch / self.warmup_iters ), self.min_lr ) for base_lr in self.base_lrs ]
        else:
            return [ max( base_lr * ( max(1 - (self.last_epoch - self.warmup_iters) / (self.max_iters - self.warmup_iters), 0) )**self.power, self.min_lr) for base_lr in self.base_lrs ]
    