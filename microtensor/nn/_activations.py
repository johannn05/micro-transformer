from typing import *
from ..core import Tensor
import cupy as np

def relu(tn: Tensor) -> Tensor:
    """
    The Rectified Linear Unit (ReLU) activation function is widely used in 
    neural networks. It is defined as:

    Formula:
    `ReLU(x) = max(0, x)`

    This function returns the input value if it is positive and 0 otherwise.

    Args:
        tn (Tensor): The input tensor.

    Returns:
        Tensor: A new tensor with ReLU applied element-wise.
    """
    # Apply the ReLU function
    out = Tensor(
        tn.data.clip(0), dtype=tn.dtype, _children=(tn,), 
        _op="relu"
    )

    if tn.requires_grad and Tensor.grad_is_enabled:
        # Backward function for ReLU
        def _relu_backward():
            tn.grad += (tn.data > 0) * out.grad

        out.grad_fn = _relu_backward
        out.set_requires_grad(True)

    return out

def tanh(tn: Tensor) -> Tensor:
    """
    Apply the hyperbolic tangent (tanh) activation function element-wise.

    Formula:
    `tanh(x) = (e^x - e^-x) / (e^x + e^-x)`
  
    Args:
        tn (Tensor): The input tensor.

    Returns:
        Tensor: A new tensor with tanh applied element-wise.
    """
    # Apply the tanh function
    out = Tensor(
        np.tanh(tn.data),     
        dtype=tn.dtype,
        _children=(tn,),
        _op="tanh"
    )

    if tn.requires_grad and Tensor.grad_is_enabled:
        # Backward function for tanh
        def _tanh_backward():
            tn.grad += (1 - out.data**2) * out.grad

        out.grad_fn = _tanh_backward
        out.set_requires_grad(True)

    return out

def sigmoid(tn: Tensor) -> Tensor:
    """
    Computes the expit (also known as the logistic sigmoid function) of the elements of input.

    Formula:
    `sigmoid(x) = 1 / (1 + exp(-x))`
    """
    exp_neg = np.exp(-tn.data)
    out = Tensor(
        1 / (1 + exp_neg), _children=(tn,), dtype=tn.dtype, _op="sigmoid"
    )

    # since d/dx (1 / (1 + e^-x)) = e^-x / (1 + e^-x) ^ 2
    if tn.requires_grad and Tensor.grad_is_enabled:
        def _sigmoid_backward():
            tn.grad += (exp_neg / (1 + exp_neg) ** 2) * out.grad

        out.grad_fn = _sigmoid_backward
        out.set_requires_grad(True)

    return out


def softmax(tn: Tensor, axis: int = -1) -> Tensor:
    """
    Apply the softmax function along a specified axis.

    Formula:
    `softmax(x)_i = exp(x_i) / sum(exp(x_j))`

    Softmax rescales the tensor elements such that they lie in the range [0, 1] 
    and sum to 1 along the specified axis.

    Args:
        tn (Tensor): The input tensor.
        axis (int, optional): The axis along which softmax is computed. Default is -1.

    Returns:
        Tensor: A new tensor with softmax applied along the specified axis.
    """
    # For numerical stability, subtract the max value from the input tensor
    max_tn = np.max(tn.data, axis=axis, keepdims=True)
    exp_shifted = np.exp(tn.data - max_tn)

    # Compute the softmax output
    exp_sum = np.sum(exp_shifted, axis=axis, keepdims=True)
    out = Tensor(
        exp_shifted / exp_sum, dtype=tn.dtype, _children=(tn,), 
        _op="softmax"
    )

    # No custom backward function is needed because it can be built from atomic operations
    return out
