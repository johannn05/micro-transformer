from typing import *

from ._module import Module
from ..core import Tensor
import cupy as np  # Using numpy for random number generation


class Dropout(Module):
    """
    randomly zero out elements during training to reduce overfitting.
    """
    def __init__(self, p: float = 0.5, device: str = "cpu") -> None:
        """
        initializes the dropout module.

        args:
            p (float): probability of an element being zeroed. default is 0.5.
            device (str): device to run the module on ('cpu' or 'gpu'). default is 'cpu'.
        """
        super().__init__(device=device)
        self.p = p
        self.is_training = True  # inherited from the base `Module` class

    def forward(self, x: Tensor) -> Tensor:
        """
        applies dropout to the input tensor during training.

        args:
            x (Tensor): input tensor.

        returns:
            Tensor: tensor with elements randomly zeroed out (during training).
        """
        # if not in training mode, return the input as is
        if not self.is_training:
            return x

        # generate a mask to randomly zero out elements
        mask = None
        # generate a mask to randomly zero out elements
        mask = np.random.binomial(n=1, p=1 - self.p, size=x.shape)


        # convert mask to Tensor
        mask = Tensor(mask, dtype=int, requires_grad=False)

        # scale the output to maintain the expected value during training
        out = x * mask / (1 - self.p)
        return out
