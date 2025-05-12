from typing import List
from ._optimizer import Optimizer
from ..core import Tensor
import cupy as np

class RMSProp(Optimizer):
    """
    implements RMSProp optimization algorithm
    """
    def __init__(
        self, 
        parameters: List[Tensor], 
        lr: float, 
        decay: float = 0.95, 
        eps: float = 1e-8
    ) -> None:
        super().__init__(parameters, lr)

        self.decay = decay
        self.eps = eps

        # initialize the running average of squared gradients
        self.vs = [np.zeros_like(p.data) for p in self.parameters]

    def step(self) -> None:
        """
        Performs a single optimization step.
        Applies adaptive learning rates to parameters using RMSProp.
        """
        for i in range(len(self.parameters)):
            p, v = self.parameters[i], self.vs[i]

            # update the running average of squared gradients
            v[:] = (self.decay * v) + ((1 - self.decay) * (p.grad ** 2))

            # update parameters
            p.data -= (self.lr / (np.sqrt(v) + self.eps)) * p.grad
