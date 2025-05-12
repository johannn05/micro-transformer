
import torch
import cupy as np
from microtensor.core import Tensor as MicroTensor
from microtensor.nn._activations import softmax
class MicrotensorAttentionOps:
    @staticmethod
    def scaled_dot_product(q, k, v, scale, mask=None):
        # Save original device for returning result
        device = q.device
        
        # Convert to microtensor
        q_micro = MicroTensor(np.array(q.detach().cpu().numpy()), requires_grad=False)
        k_micro = MicroTensor(np.array(k.detach().cpu().numpy()), requires_grad=False)
        v_micro = MicroTensor(np.array(v.detach().cpu().numpy()), requires_grad=False)
        
        # Compute attention scores
        k_t = k_micro.T(axes=[0, 1, 3, 2])  # Transpose last two dimensions
        scores = q_micro @ k_t
        
        # Apply scaling
        scores = scores * scale
        
        # Apply mask if provided
        if mask is not None:
            # Convert mask to CuPy array directly (not MicroTensor)
            mask_np = np.array(mask.detach().cpu().numpy())
            
            # Use np.where directly on the data
            scores_data = np.where(mask_np == 0, np.full_like(scores.data, -1e9), scores.data)
            
            # Create a new MicroTensor with the masked data
            scores = MicroTensor(scores_data, dtype=scores.dtype, requires_grad=False)
        
        # Apply softmax
        from microtensor.nn._activations import softmax
        attn_weights = softmax(scores, axis=-1)
        
        # Apply attention to values
        output = attn_weights @ v_micro
        
        # Convert back to PyTorch tensor
        return torch.tensor(output.data, device=device)
    
class MicrotensorMultiHeadAttention:
    @staticmethod
    def apply_attention(q_proj: torch.Tensor, k_proj: torch.Tensor, v_proj: torch.Tensor, 
                        mask: torch.Tensor = None, head_dim: int = 64) -> torch.Tensor:
        """
        Apply multi-head attention using microtensor
        
        Args:
            q_proj: Projected queries [batch, seq_len_q, heads*head_dim]
            k_proj: Projected keys [batch, seq_len_k, heads*head_dim]
            v_proj: Projected values [batch, seq_len_k, heads*head_dim]
            mask: Optional attention mask [batch, 1, seq_len_q, seq_len_k]
            head_dim: Dimension of each attention head
            
        Returns:
            Output tensor [batch, seq_len_q, heads*head_dim]
        """
        batch_size, seq_len_q, d_model = q_proj.shape
        batch_size, seq_len_k, _ = k_proj.shape
        
        # Calculate number of heads
        num_heads = d_model // head_dim
        
        # Reshape projected tensors to separate heads
        q_reshaped = q_proj.view(batch_size, seq_len_q, num_heads, head_dim)
        k_reshaped = k_proj.view(batch_size, seq_len_k, num_heads, head_dim)
        v_reshaped = v_proj.view(batch_size, seq_len_k, num_heads, head_dim)
        
        # Transpose to [batch, heads, seq_len, head_dim]
        q_transposed = q_reshaped.transpose(1, 2)
        k_transposed = k_reshaped.transpose(1, 2)
        v_transposed = v_reshaped.transpose(1, 2)
        
        # Calculate scaling factor
        scale = 1.0 / np.sqrt(head_dim)
        
        # Apply scaled dot-product attention
        attn_output = MicrotensorAttentionOps.scaled_dot_product(
            q_transposed, k_transposed, v_transposed, scale, mask
        )
        
        # Transpose back to [batch, seq_len, heads, head_dim]
        output_transposed = attn_output.transpose(1, 2)
        
        # Reshape to [batch, seq_len, d_model]
        output = output_transposed.reshape(batch_size, seq_len_q, d_model)
        
        return output