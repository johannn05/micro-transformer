from typing import List
from ._optimizer import Optimizer
from ..core import Tensor
import cupy as np

class Adam(Optimizer):
    """
    Adam optimizer implementation.
    """
    def __init__(self, parameters: List[Tensor], lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.moments1 = [np.zeros_like(p.data) for p in self.parameters]
        self.moments2 = [np.zeros_like(p.data) for p in self.parameters]
        self.timestep = 0

    def step(self):
        self.timestep += 1
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            self.moments1[i] = self.beta1 * self.moments1[i] + (1 - self.beta1) * p.grad
            self.moments2[i] = self.beta2 * self.moments2[i] + (1 - self.beta2) * (p.grad ** 2)

            m_hat = self.moments1[i] / (1 - self.beta1 ** self.timestep)
            v_hat = self.moments2[i] / (1 - self.beta2 ** self.timestep)

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
