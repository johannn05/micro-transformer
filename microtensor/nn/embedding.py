from typing import Optional
from ..core import Tensor
from ._module import Module
import cupy as np
import math

class Embedding(Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        scale_grad_by_freq: bool = False,
        device: str = "cpu"
    ):
        """
        Initialize an embedding layer with enhanced functionality.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: If given, embeddings at this index will not be updated
            max_norm: If given, embeddings will be normalized to have this maximum norm
            scale_grad_by_freq: If True, scale gradients by frequency of tokens
            device: Computation device
        """
        super().__init__(device)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.scale_grad_by_freq = scale_grad_by_freq

        # Initialize embeddings using Xavier/Glorot initialization
        std = 1.0 / math.sqrt(embedding_dim)
        self.weight = Tensor(
            np.random.normal(0, std, (num_embeddings, embedding_dim)).astype(np.float32),
            requires_grad=True
        )

        # Initialize padding tokens to zero if padding_idx is specified
        if self.padding_idx is not None:
            # Vectorized zeroing of the padding row(s):
            self.weight.data[self.padding_idx] = np.zeros_like(
                self.weight.data[self.padding_idx]
    )


    def forward(self, input: Tensor) -> Tensor:
        """
        Perform embedding lookup with proper gradient computation.
        
        Args:
            input: Tensor of token indices
                
        Returns:
            Tensor of embeddings
        """
        # Convert input to integer type if needed
        if input.dtype not in [np.int32, np.int64]:
            input.data = input.data.astype(np.int32)
            input.dtype = np.int32

        # Check input values are within valid range
        if np.any((input.data < 0) | (input.data >= self.num_embeddings)):
            raise ValueError(
                f"Input values must be between 0 and {self.num_embeddings-1}"
            )

        # Perform embedding lookup
        out = Tensor(
            self.weight.data[input.data],
            dtype=self.weight.dtype,
            _children=(self.weight, input),  # Include both weight and input in computation graph
            _op="embedding",
            requires_grad=True  # Always enable gradient tracking for embeddings
        )

        # Set up gradient computation
        if self.weight.requires_grad and Tensor.grad_is_enabled:
            def _embedding_backward():
                # Initialize gradient tensor
                grad = np.zeros_like(self.weight.data)
                
                # Flatten input and output gradient for easier indexing
                flat_indices = input.data.reshape(-1)
                flat_grad = out.grad.reshape(-1, self.embedding_dim)
                
                # Accumulate gradients for each index
                for idx, grad_val in zip(flat_indices, flat_grad):
                    grad[idx] += grad_val
                    
                # Scale gradients if using frequency-based scaling
                if self.scale_grad_by_freq:
                    counts = np.bincount(
                        flat_indices,
                        minlength=self.num_embeddings
                    )
                    scale = 1.0 / np.maximum(1.0, counts[flat_indices])
                    grad *= scale.reshape(-1, 1)

                # Zero gradients for padding index
                if self.padding_idx is not None:
                    grad[self.padding_idx] = 0

                # Apply max norm constraint
                if self.max_norm is not None:
                    norms = np.linalg.norm(self.weight.data, axis=1)
                    scale = np.minimum(self.max_norm / norms, 1.0)
                    grad *= scale[:, None]

                # Update gradients
                self.weight.grad += grad

            out.grad_fn = _embedding_backward

        return out
    def extra_repr(self) -> str:
        """Returns a string with extra information about the layer."""
        return (f'num_embeddings={self.num_embeddings}, '
                f'embedding_dim={self.embedding_dim}, '
                f'padding_idx={self.padding_idx}')