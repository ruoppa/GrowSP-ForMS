import torch.nn as nn

from easydict import EasyDict
from .build import LOSSES


@LOSSES.register_module()
class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, config: EasyDict) -> None:
        try:
            super().__init__(**config.kwargs)
        except TypeError as e:
            raise TypeError("Error while building loss from config!") from e
