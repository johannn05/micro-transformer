import math
from typing import *
from ..core import Tensor
from ._module import Module
from .linear import Linear
from .dropout import Dropout
from ._activations import softmax
import cupy as np

class MultiHeadAttention(Module):
    """
    Base class for multi-head attention mechanisms.
    Implements the core scaled dot-product attention logic.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__(device)
        
        # Ensure d_model is divisible by n_heads
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Dimension of each head
        
        # Linear projections
        self.q_proj = Linear(d_model, d_model, device=device)
        self.k_proj = Linear(d_model, d_model, device=device)
        self.v_proj = Linear(d_model, d_model, device=device)
        self.o_proj = Linear(d_model, d_model, device=device)
        
        self.dropout = Dropout(dropout, device=device)
        self.scale = 1.0 / math.sqrt(self.d_k)  # Scaling factor for attention scores
    
    def _compute_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute scaled dot-product attention.
        """
        # Compute attention scores
        attn_scores = (q @ k.T(axes=[0, 1, 3, 2])) * self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        # Note: Using softmax function directly from _activations
        attn_weights = self.dropout(softmax(attn_scores, axis=-1))
        
        # Compute output
        attn_output = attn_weights @ v
        
        # Ensure gradient tracking is maintained
        attn_output.requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
        attn_weights.requires_grad = q.requires_grad or k.requires_grad
        
        return attn_output, attn_weights
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply multi-head attention.
        """
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        
        # Print debug info
        print("\nAttention Forward Debug:")
        print(f"Query requires_grad: {query.requires_grad}")
        print(f"Key requires_grad: {key.requires_grad}")
        print(f"Value requires_grad: {value.requires_grad}")
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        print(f"After projections:")
        print(f"Q requires_grad: {q.requires_grad}")
        print(f"K requires_grad: {k.requires_grad}")
        print(f"V requires_grad: {v.requires_grad}")
        
        # Reshape projections
        q = q.reshape((batch_size, seq_len_q, self.n_heads, self.d_k)).T(axes=[0, 2, 1, 3])
        k = k.reshape((batch_size, seq_len_k, self.n_heads, self.d_k)).T(axes=[0, 2, 1, 3])
        v = v.reshape((batch_size, seq_len_k, self.n_heads, self.d_k)).T(axes=[0, 2, 1, 3])
        
        print(f"After reshape:")
        print(f"Q requires_grad: {q.requires_grad}")
        print(f"K requires_grad: {k.requires_grad}")
        print(f"V requires_grad: {v.requires_grad}")
        
        # Compute attention
        attn_output, attn_weights = self._compute_attention(q, k, v, mask)
        
        print(f"After attention:")
        print(f"Output requires_grad: {attn_output.requires_grad}")
        
        # Reshape and project output
        attn_output = (attn_output.T(axes=[0, 2, 1, 3])
                    .reshape((batch_size, seq_len_q, self.d_model)))
        
        output = self.o_proj(attn_output)
        
        print(f"Final output requires_grad: {output.requires_grad}")
        
        # Maintain gradient tracking
        requires_grad = (query.requires_grad or key.requires_grad or value.requires_grad or 
                        any(p.requires_grad for p in self.parameters()))
        
        output.requires_grad = requires_grad
        attn_weights.requires_grad = requires_grad
        
        print(f"After final adjustment - output requires_grad: {output.requires_grad}")
        
        return output, attn_weights


class CausalSelfAttention(MultiHeadAttention):
    """
    Causal Self-Attention module that prevents attending to future tokens.
    Extends MultiHeadAttention with causal masking.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.1,
        device: str = "cpu"
    ):
        super().__init__(d_model, n_heads, dropout, device)
        
        # Create causal mask once during initialization
        mask = np.triu(np.ones((max_seq_len, max_seq_len)), k=1)
        self.register_buffer(
            'causal_mask',
            Tensor(mask == 0, dtype=np.bool_, requires_grad=False)
        )
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Apply causal self-attention to input sequence.
        """
        # Create causal mask for current sequence length
        seq_len = x.shape[1]
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        
        # Combine with padding mask if provided
        if mask is not None:
            causal_mask = causal_mask & mask
        
        # Apply self-attention with causal mask
        output, _ = super().forward(x, x, x, causal_mask)
        
        # Maintain gradient tracking
        output.requires_grad = x.requires_grad or any(p.requires_grad for p in self.parameters())
        
        return output