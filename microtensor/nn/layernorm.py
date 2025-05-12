from typing import Union, List, Tuple
from ..core import Tensor
from ._module import Module
import cupy as np

class LayerNorm(Module):
    """
    Layer Normalization module that normalizes input tensors along specified dimensions.
    
    Layer normalization helps stabilize deep networks by normalizing activations
    across features rather than batch dimension. This is particularly important
    for transformer architectures where batch statistics can be unreliable due
    to variable sequence lengths.
    """
    def __init__(
        self,
        normalized_shape: Union[int, List[int], Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: str = "cpu"
    ):
        """
        Initialize LayerNorm module.
        
        Args:
            normalized_shape: Shape of the features to normalize over. Can be:
                - Single integer for 1D inputs
                - List/Tuple of integers for ND inputs
            eps: Small constant for numerical stability
            elementwise_affine: Whether to include learnable affine parameters
            device: Computation device
        """
        super().__init__(device)
        
        # Convert normalized_shape to tuple for consistency
        if isinstance(normalized_shape, (list, tuple)):
            self.normalized_shape = tuple(normalized_shape)
        else:
            self.normalized_shape = (normalized_shape,)
            
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        # Initialize affine parameters if needed
        if elementwise_affine:
            self.weight = Tensor(
                np.ones(self.normalized_shape, dtype=np.float32),
                requires_grad=True
            )
            self.bias = Tensor(
                np.zeros(self.normalized_shape, dtype=np.float32),
                requires_grad=True
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to input tensor.
        
        Args:
            x: Input tensor of shape (*, normalized_shape)
                where * represents any number of leading dimensions
                
        Returns:
            Normalized tensor of the same shape
        """
        # Check for environment variable to disable microtensor
        import os
        if os.environ.get('DISABLE_MICROTENSOR') == '1':
            # Use standard implementation without microtensor-specific code
            # Validate input shape
            if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
                raise ValueError(
                    f"Expected normalized_shape {self.normalized_shape}, "
                    f"got input shape {x.shape}"
                )

            # Calculate normalization dimensions
            norm_dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
            
            # Calculate statistics for normalization using standard numpy operations
            import numpy
            x_numpy = x.data.get()  # Convert to numpy array
            
            mean = numpy.mean(x_numpy, axis=norm_dims, keepdims=True)
            var = numpy.mean((x_numpy - mean) ** 2, axis=norm_dims, keepdims=True)
            std = numpy.sqrt(var + self.eps)

            # Normalize the input
            x_norm = (x_numpy - mean) / std

            # Apply affine transformation if enabled
            if self.elementwise_affine:
                weight_numpy = self.weight.data.get()
                bias_numpy = self.bias.data.get()
                out_data = weight_numpy * x_norm + bias_numpy
            else:
                out_data = x_norm
                
            # Return as normal tensor without tracking gradients
            return Tensor(out_data, requires_grad=False)
        else:
            # Original microtensor implementation
            # Validate input shape
            if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
                raise ValueError(
                    f"Expected normalized_shape {self.normalized_shape}, "
                    f"got input shape {x.shape}"
                )

            # Calculate normalization dimensions
            norm_dims = tuple(range(len(x.shape) - len(self.normalized_shape), len(x.shape)))
            
            # Calculate statistics for normalization
            mean = x.data.mean(axis=norm_dims, keepdims=True)
            var = ((x.data - mean) ** 2).mean(axis=norm_dims, keepdims=True)
            std = np.sqrt(var + self.eps)

            # Normalize the input
            x_norm = (x.data - mean) / std

            # Apply affine transformation if enabled
            if self.elementwise_affine:
                out_data = self.weight.data * x_norm + self.bias.data
                children = (x, self.weight, self.bias)
            else:
                out_data = x_norm
                children = (x,)

            # Create output tensor
            out = Tensor(
                out_data,
                dtype=x.dtype,
                _children=children,
                _op="layernorm",
                requires_grad=x.requires_grad or (self.elementwise_affine and (self.weight.requires_grad or self.bias.requires_grad))
            )

            if x.requires_grad and Tensor.grad_is_enabled:
                def _layernorm_backward():
                    # Number of elements being normalized
                    N = np.prod([x.shape[dim] for dim in norm_dims])
                    
                    if self.elementwise_affine:
                        # Gradient with respect to normalized input
                        grad_norm = out.grad * self.weight.data
                    else:
                        grad_norm = out.grad

                    # Compute gradient for input
                    mean_grad = grad_norm.mean(axis=norm_dims, keepdims=True)
                    var_grad = ((grad_norm * x_norm) * -0.5 / std).mean(
                        axis=norm_dims, keepdims=True
                    )
                    
                    grad_input = (
                        grad_norm / std - 
                        mean_grad / std - 
                        2 * var_grad * (x.data - mean) / N
                    )
                    
                    x.grad += grad_input

                    # Compute gradients for affine parameters if used
                    if self.elementwise_affine:
                        self.weight.grad += (out.grad * x_norm).sum(
                            axis=tuple(range(len(x.shape) - len(self.normalized_shape)))
                        )
                        self.bias.grad += out.grad.sum(
                            axis=tuple(range(len(x.shape) - len(self.normalized_shape)))
                        )

                out.grad_fn = _layernorm_backward
                out.set_requires_grad(True)

            return out

    def extra_repr(self) -> str:
        """Return a string with extra information about the layer."""
        return (f'normalized_shape={self.normalized_shape}, '
                f'eps={self.eps}, '
                f'elementwise_affine={self.elementwise_affine}')