import torch
import torch.nn as nn
import math
from utils.layers import MicrotensorLayerNorm
from utils.attention import MicrotensorAttentionOps


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # Multiply by sqrt(d_model) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_model)
        return self.dropout(x)

class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) # learnable parameter
        self.bias = nn.Parameter(torch.zeros(features)) # learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Check for environment variable to disable microtensor
        import os
        if os.environ.get('DISABLE_MICROTENSOR') == '1':
            # Use original PyTorch implementation
            return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
        else:
            # Use microtensor implementation
            from utils.feedforward import MicrotensorFeedForward
            return MicrotensorFeedForward.forward(
                x, 
                self.linear_1.weight, 
                self.linear_1.bias,
                self.linear_2.weight,
                self.linear_2.bias,
                dropout_rate=self.dropout.p,
                training=self.training
            )
    
class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = MicrotensorLayerNorm(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))

class MultiHeadLatentAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, d_c: int, d_cr: int, d_rope: int, dropout: float) -> None:
        super().__init__()
        self.h = h
        self.d_k = d_model // h
        self.d_c = d_c
        self.d_cr = d_cr
        self.d_rope = d_rope

        self.W_DKV = nn.Linear(d_model, d_c)
        self.W_UK = nn.Linear(d_c, d_model)
        self.W_UV = nn.Linear(d_c, d_model)

        self.W_DQ = nn.Linear(d_model, d_cr)
        self.W_UQ = nn.Linear(d_cr, d_model)

        self.W_QR = nn.Linear(d_cr, h * d_rope)
        self.W_KR = nn.Linear(d_model, d_rope)

        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        # Check for environment variable to disable microtensor
        import os
        if os.environ.get('DISABLE_MICROTENSOR') == '1':
            # Original PyTorch implementation
            if not (q is k and k is v):
                # Apply projection for cross attention
                c_kv = self.W_DKV(k)
                k_C = self.W_UK(c_kv)
                v_C = self.W_UV(c_kv)

                c_q = self.W_DQ(q)
                q_C = self.W_UQ(c_q)

                q_R = self.W_QR(c_q).view(q.shape[0], q.shape[1], self.h, self.d_rope)
                k_R = self.W_KR(k).unsqueeze(2).expand(-1, -1, self.h, -1)
                if not hasattr(self, "_debugged"):
                    print("q_C mean/std:", q_C.mean().item(), q_C.std().item())
                    print("k_C mean/std:", k_C.mean().item(), k_C.std().item())
                    print("q_R mean/std:", q_R.mean().item(), q_R.std().item())
                    print("k_R mean/std:", k_R.mean().item(), k_R.std().item())
                    self._debugged = True
            else:
                x = q
                c_kv = self.W_DKV(x)
                k_C = self.W_UK(c_kv)
                v_C = self.W_UV(c_kv)

                c_q = self.W_DQ(x)
                q_C = self.W_UQ(c_q)

                q_R = self.W_QR(c_q).view(x.shape[0], x.shape[1], self.h, self.d_rope)
                k_R = self.W_KR(x).unsqueeze(2).expand(-1, -1, self.h, -1)
            
            # Use original PyTorch rope implementation
            half = q_R.shape[-1] // 2
            freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32, device=q_R.device) / half))
            angle = q_R[..., :half] * freq.view(1, 1, 1, -1)
            q_R = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
            
            angle = k_R[..., :half] * freq.view(1, 1, 1, -1)
            k_R = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)

            B, T_q = q_C.shape[:2]
            T_k = k_C.shape[1]

            q_C_split = q_C.view(B, T_q, self.h, self.d_k)
            k_C_split = k_C.view(B, T_k, self.h, self.d_k)

            q_C_base = q_C_split[:, :, :, :self.d_k - self.d_rope]
            k_C_base = k_C_split[:, :, :, :self.d_k - self.d_rope]

            q = torch.cat([q_C_base, q_R], dim=-1)
            k = torch.cat([k_C_base, k_R], dim=-1)
            v = v_C.view(B, T_k, self.h, self.d_k)

            # Original attention computation
            q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            attn = self.dropout(scores.softmax(dim=-1))
            x = attn @ v
            x = x.transpose(1, 2).contiguous().view(B, T_q, -1)
            
            return self.W_O(x)
        else:
            # Microtensor implementation
            if not (q is k and k is v):
                # Apply projection for cross attention
                c_kv = self.W_DKV(k)
                k_C = self.W_UK(c_kv)
                v_C = self.W_UV(c_kv)

                c_q = self.W_DQ(q)
                q_C = self.W_UQ(c_q)

                q_R = self.W_QR(c_q).view(q.shape[0], q.shape[1], self.h, self.d_rope)
                k_R = self.W_KR(k).unsqueeze(2).expand(-1, -1, self.h, -1)
                if not hasattr(self, "_debugged"):
                    print("q_C mean/std:", q_C.mean().item(), q_C.std().item())
                    print("k_C mean/std:", k_C.mean().item(), k_C.std().item())
                    print("q_R mean/std:", q_R.mean().item(), q_R.std().item())
                    print("k_R mean/std:", k_R.mean().item(), k_R.std().item())
                    self._debugged = True
            else:
                x = q
                c_kv = self.W_DKV(x)
                k_C = self.W_UK(c_kv)
                v_C = self.W_UV(c_kv)

                c_q = self.W_DQ(x)
                q_C = self.W_UQ(c_q)

                q_R = self.W_QR(c_q).view(x.shape[0], x.shape[1], self.h, self.d_rope)
                k_R = self.W_KR(x).unsqueeze(2).expand(-1, -1, self.h, -1)
            
            q_R = self.apply_rope(q_R)
            k_R = self.apply_rope(k_R)

            B, T_q = q_C.shape[:2]
            T_k = k_C.shape[1]

            q_C_split = q_C.view(B, T_q, self.h, self.d_k)
            k_C_split = k_C.view(B, T_k, self.h, self.d_k)

            q_C_base = q_C_split[:, :, :, :self.d_k - self.d_rope]
            k_C_base = k_C_split[:, :, :, :self.d_k - self.d_rope]

            q = torch.cat([q_C_base, q_R], dim=-1)
            k = torch.cat([k_C_base, k_R], dim=-1)
            v = v_C.view(B, T_k, self.h, self.d_k)

            # Use microtensor attention
            from utils.attention import MicrotensorAttentionOps
            x = MicrotensorAttentionOps.scaled_dot_product(
                q.transpose(1, 2),  # [B, H, T_q, dim]
                k.transpose(1, 2),  # [B, H, T_k, dim]
                v.transpose(1, 2),  # [B, H, T_k, dim]
                scale=1.0/math.sqrt(self.d_k),
                mask=mask
            )

            # Reshape the output to match the original format
            x = x.transpose(1, 2).contiguous().view(B, T_q, -1)
            return self.W_O(x)
    

    def apply_rope(self, x):
        half = x.shape[-1] // 2
        freq = 1.0 / (10000 ** (torch.arange(half, dtype=torch.float32, device=x.device) / half))
        angle = x[..., :half] * freq.view(1, 1, 1, -1)  # broadcast to B, T, H, half
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)



class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadLatentAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = MicrotensorLayerNorm(features)


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadLatentAttentionBlock, cross_attention_block: MultiHeadLatentAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = MicrotensorLayerNorm(features)


    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq_len, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    def forward(self, src, src_mask, tgt, tgt_mask):
        # First run through the encoder
        encoder_output = self.encode(src, src_mask)
        # Then run through the decoder
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)
        # Finally project to get the output
        return self.project(decoder_output)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    # create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadLatentAttentionBlock(
    d_model=d_model,
    h=h,
    d_c=512,
    d_cr=1536,
    d_rope = 32,
    dropout=dropout
)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadLatentAttentionBlock(
    d_model=d_model,
    h=h,
    d_c=512,
    d_cr=1536,
    d_rope=32,
    dropout=dropout
)

        decoder_cross_attention_block = MultiHeadLatentAttentionBlock(
    d_model=d_model,
    h=h,
    d_c=512,
    d_cr=1536,
    d_rope=32,
    dropout=dropout
)

        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # create  projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer

def make_model_quantizable(model):
    """
    Add quantization support to model
    """
    # Replace layer implementations with quantizable versions
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            model._modules[name] = nn.quantized.Linear.from_float(module)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm is not quantized
            pass
        else:
            make_model_quantizable(module)
    
    return model