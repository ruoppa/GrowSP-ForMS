import torch.optim as optim
from easydict import EasyDict
from .build import SCHEDULERS


class LambdaStepLR(optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        # Use last_epoch for the step counter
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


@SCHEDULERS.register_module()
class PolyLR(LambdaStepLR):
    # DeepLab learning rate policy
    def __init__(self, config: EasyDict, optimizer: optim.Optimizer):
        # All scheduler parameters that can be defined in the config also have a default value
        max_iter = config.kwargs.get("max_iter", 30000)
        power = config.kwargs.get("power", 0.9)
        last_step = config.kwargs.get("last_step", -1)
        super(PolyLR, self).__init__(
            optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step
        )


@SCHEDULERS.register_module()
class OneCycleLR(optim.lr_scheduler.OneCycleLR):
    def __init__(self, config: EasyDict, optimizer: optim.Optimizer):
        try:
            super().__init__(optimizer, **config.kwargs)
        except TypeError as e:
            raise TypeError("Error while building optimizer from config!") from e
