from typing import List
from ..core import Tensor

class Optimizer:
    """
    base class for optimizers
    """
    def __init__(self, parameters: List[Tensor], lr: float) -> None:
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        """
        update params
        """
        raise NotImplementedError("The 'step' method must be implemented in the child class.")

    def zero_grad(self) -> None:
        """
        reset gradients to zero
        """
        for param in self.parameters:
            if param.grad is not None:
                param.grad.fill(0)
