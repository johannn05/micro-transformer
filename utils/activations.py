import cupy as np
from microtensor.nn._activations import relu, softmax
import torch

class MicrogradActivation:
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        # Convert PyTorch tensor to CuPy array
        x_cupy = np.array(x.detach().cpu().numpy())
        # Apply micrograd relu
        from microtensor.core import Tensor
        x_tensor = Tensor(x_cupy, requires_grad=False)
        result_tensor = relu(x_tensor)
        # Convert back to PyTorch tensor
        return torch.tensor(result_tensor.data, device=x.device)
    
    @staticmethod
    def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Convert PyTorch tensor to CuPy array
        x_cupy = np.array(x.detach().cpu().numpy())
        # Apply micrograd softmax
        from microtensor.core import Tensor
        x_tensor = Tensor(x_cupy, requires_grad=False)
        result_tensor = softmax(x_tensor, axis=dim)
        # Convert back to PyTorch tensor
        return torch.tensor(result_tensor.data, device=x.device)