import torch.optim as optim
import torch.nn as nn

from easydict import EasyDict
from .build import OPTIMIZERS


@OPTIMIZERS.register_module()
class SGD(optim.SGD):
    def __init__(self, config: EasyDict, base_model: nn.Module) -> None:
        try:
            super().__init__(base_model.parameters(), **config.kwargs)
        except TypeError as e:
            raise TypeError("Error while building optimizer from config!") from e


@OPTIMIZERS.register_module()
class AdamW(optim.AdamW):
    def __init__(self, config: EasyDict, base_model: nn.Module) -> None:
        try:
            super().__init__(base_model.parameters(), **config.kwargs)
        except TypeError as e:
            raise TypeError("Error while building optimizer from config!") from e


"""
Adding other supported optimizers is very straightforward (provided you wish to use some default pytorch optimizer).
Simply copy and paste the above and replace SGD with optimizer of your choice. Remember to also register the optimizer
to the optimizer registry
"""
