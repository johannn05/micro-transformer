from ..core import Tensor
from ._module import Module
import cupy as np

class Linear(Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        use_bias: bool = True, 
        device: str = "cpu",
        dtype=None
    ):
        """
        Fully connected linear layer with proper initialization.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to include bias terms
            device: Device for computation ('cpu' only currently supported)
            dtype: Data type for the tensors
        """
        # Initialize parent class
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.device = device
        self.dtype = dtype  # Add this line

        # He initialization for better gradient flow
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            dtype=dtype,
            requires_grad=True
        )

        if self.use_bias:
            self.bias = Tensor(
                np.zeros(out_features),
                dtype=dtype,
                requires_grad=True
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer with proper gradient computation.
        """
        # Store the input for backward pass
        self.saved_input = x
        original_shape = x.shape
        input_is_3d = len(original_shape) > 2
        
        # Reshape to 2D if needed
        if input_is_3d:
            x = x.reshape(-1, self.in_features)
        
        # Compute output using numpy arrays
        out_data = x.data @ self.weight.data.T
        if self.use_bias:
            out_data = out_data + self.bias.data
        
        # Create initial output tensor (in 2D form)
        should_track_grad = x.requires_grad or self.weight.requires_grad or (self.use_bias and self.bias.requires_grad)
        
        out = Tensor(
            out_data,
            dtype=x.dtype,
            _children=(self.weight, x) + ((self.bias,) if self.use_bias else ()),
            _op="linear",
            requires_grad=should_track_grad
        )
        
        # Reshape back to 3D if input was 3D
        if input_is_3d:
            out = out.reshape(*original_shape[:-1], self.out_features)
        
        if should_track_grad and Tensor.grad_is_enabled:
            def _linear_backward():
                if out.grad is None:
                    return
                    
                # Get output gradient and reshape to 2D if needed
                out_grad = out.grad
                if input_is_3d:
                    out_grad_2d = out.grad.reshape(-1, self.out_features)
                    x_2d = self.saved_input.data.reshape(-1, self.in_features)
                else:
                    out_grad_2d = out.grad
                    x_2d = self.saved_input.data
                
                # Compute weight gradients
                if self.weight.requires_grad:
                    if self.weight.grad is None:
                        self.weight.grad = np.zeros_like(self.weight.data)
                    grad_weight = out_grad_2d.T @ x_2d
                    self.weight.grad += grad_weight
                
                # Compute bias gradients
                if self.use_bias and self.bias.requires_grad:
                    if self.bias.grad is None:
                        self.bias.grad = np.zeros_like(self.bias.data)
                    grad_bias = out_grad_2d.sum(axis=0)
                    self.bias.grad += grad_bias
                
                # Compute input gradients
                if x.requires_grad:
                    # Initialize x.grad with the original input shape
                    if x.grad is None:
                        x.grad = np.zeros_like(self.saved_input.data)  # This will have the correct shape
                    
                    grad_input = out_grad_2d @ self.weight.data
                    
                    # Reshape grad_input to match original input shape
                    if input_is_3d:
                        grad_input = grad_input.reshape(original_shape)
                    
                    # Now both shapes will match
                    self.saved_input.grad = self.saved_input.grad + grad_input if self.saved_input.grad is not None else grad_input
                    x.grad = self.saved_input.grad  # Ensure x.grad points to the same gradient
            
            out.grad_fn = _linear_backward
        
        return out
    def extra_repr(self) -> str:
        """
        Returns a string with extra information about the layer.
        Useful for debugging and model inspection.
        """
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.use_bias}'