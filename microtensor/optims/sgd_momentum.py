from typing import List
from ..core import Tensor
from ._optimizer import Optimizer
import cupy as np

class SGD(Optimizer):
    """
    stochastic gradient descent (SGD) with momentum.
    """
    def __init__(
        self, 
        parameters: List[Tensor], 
        lr: float, 
        momentum: float = 0.9
    ) -> None:
        super().__init__(parameters, lr)

        self.momentum = momentum

        # initialize velocity for each param
        self.velocities = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        """
        perform a single optimization step using momentum

        velocity and parameter update rule:
        ```
        v = momentum * v + (1 - momentum) * p.grad
        p.data -= lr * v
        ```
        """
        for i in range(len(self.parameters)):
            p, v = self.parameters[i], self.velocities[i]

            #Update velocity
            v[:] = (self.momentum * v) + ((1 - self.momentum) * p.grad)

            #Update param
            p.data -= self.lr * v
