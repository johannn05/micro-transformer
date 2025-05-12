import torch
import cupy as np
from microtensor.core import Tensor as MicroTensor

class MicrotensorFeedForward:
    @staticmethod
    def forward(x_torch, w1, b1, w2, b2, dropout_rate=0.1, training=True):
        """
        Apply feed-forward network using microtensor with CuPy backend
        
        Args:
            x_torch: Input tensor (PyTorch)
            w1: First linear layer weights
            b1: First linear layer bias
            w2: Second linear layer weights
            b2: Second linear layer bias
            dropout_rate: Probability of dropping elements (default: 0.1)
            training: Whether in training mode (apply dropout) or not
            
        Returns:
            PyTorch tensor with the result
        """
        device = x_torch.device
        
        # Convert to microtensor with CuPy backend
        x = MicroTensor(np.array(x_torch.detach().cpu().numpy()), requires_grad=False)
        
        # First linear layer
        h = x @ np.array(w1.T.detach().cpu().numpy()) + np.array(b1.detach().cpu().numpy())
        
        # ReLU
        from microtensor.nn._activations import relu
        h = relu(h)
        
        # Apply dropout if in training mode
        if training and dropout_rate > 0:
            # Create dropout mask (1 = keep, 0 = drop)
            mask = np.random.binomial(1, 1-dropout_rate, h.data.shape).astype(np.float32)
            # Apply mask and scale
            h = MicroTensor(h.data * mask / (1 - dropout_rate), requires_grad=False)
        
        # Second linear layer
        out = h @ np.array(w2.T.detach().cpu().numpy()) + np.array(b2.detach().cpu().numpy())
        
        # Convert back to PyTorch tensor
        numpy_data = out.data.get()
        result = torch.tensor(numpy_data, device=device)
        
        # Clean up to help with memory management
        del x
        del h
        del out
        del numpy_data
        
        return result