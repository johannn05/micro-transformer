from typing import *

from ..core import Tensor
from ._module import Module
from ._activations import *


class ReLU(Module):
    """
    relu module returns the input value if it is positive, 
    and 0 if the input value is negative or zero.
    """
    def __init__(self) -> None:
        super().__init__()  # ensure proper superclass initialization

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


class Sigmoid(Module):
    """
    sigmoid module computes the sigmoid activation of the input tensor.
    """
    def __init__(self) -> None:
        super().__init__()  # ensure proper superclass initialization

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class Softmax(Module):
    """
    softmax module computes the softmax function for an input tensor 
    along a specified axis.
    """
    def __init__(self, dim: Optional[int] = None) -> None:
        super().__init__()  # ensure proper superclass initialization
        self.dim = dim or -1  # default to the last dimension if none

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, axis=self.dim)


class GELU(Module):
    """
    gelu module applies the gaussian error linear unit activation function 
    with an optional approximation using tanh.

    args:
        approximate (literal['none', 'tanh']): whether to use an exact or tanh-based approximation.
        device (str): specifies the device ('gpu' or 'cpu').
    """
    def __init__(
        self, 
        approximate: Literal['none', 'tanh'] = 'none', 
        device: str = "gpu"
    ) -> None:
        super().__init__(device)  # proper superclass initialization
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        """
        compute the gelu activation based on the selected approximation mode.
        """
        if self.approximate == 'tanh':
            return self._tanh_approximation(x)
        return self._exact_gelu(x)

    def _exact_gelu(self, x: Tensor) -> Tensor:
        """
        compute the exact gelu activation using the formula:
        0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
        """
        y: Tensor = 0.7978845608 * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1 + tanh(y))

    def _tanh_approximation(self, x: Tensor) -> Tensor:
        """
        compute the gelu activation using a tanh-based approximation:
        0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
        """
        y: Tensor = (0.7978845608 * (x + 0.044715 * x * x * x))
        return 0.5 * x * (1 + tanh(y))
