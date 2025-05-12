# In utils/layers.py
import torch
import cupy as np
from microtensor.nn.layernorm import LayerNorm as MicroLayerNorm
from microtensor.core import Tensor as MicroTensor

class MicrotensorLayerNorm(torch.nn.Module):
    def __init__(self, features: int, eps: float = 1e-5):
        super().__init__()
        # Create PyTorch parameters that will be updated during backprop
        self.alpha = torch.nn.Parameter(torch.ones(features))
        self.bias = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps
        
        # Create the microtensor LayerNorm (which won't be trained directly)
        self.micro_layernorm = MicroLayerNorm(
            normalized_shape=features,
            eps=eps,
            elementwise_affine=True
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Copy latest parameters to microtensor
        self.micro_layernorm.weight.data = np.array(self.alpha.detach().cpu().numpy())
        self.micro_layernorm.bias.data = np.array(self.bias.detach().cpu().numpy())
        
        # Convert PyTorch tensor to microtensor
        x_micro = MicroTensor(
            np.array(x.detach().cpu().numpy()),
            requires_grad=False
        )
        
        # Apply microtensor layernorm
        result_micro = self.micro_layernorm(x_micro)
        
        # Convert back to PyTorch tensor
        result_torch = torch.tensor(
            result_micro.data,
            device=x.device
        )
        
        return result_torch