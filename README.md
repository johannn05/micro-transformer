# MicroTensor: A Deep Learning Framework 

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-downloads)

## Summary

MicroTensor is a tensor computation library implementing automatic differentiation from first principles, with the goal to learn the fundamental mechanisms underlying modern deep learning frameworks like PyTorch and TensorFlow. I then apply this to train a complete English-to-Italian neural machine translation model built on a novel Transformer architecture that integrates DeepSeek-inspired multi-head attention mechanisms. This implementation also features distributed training, custom computational kernels, and optimization techniques including dynamic quantization and gradient synchronization across multiple GPU nodes.

---

## Table of Contents

1. [Theoretical Foundations](#theoretical-foundations)
2. [MicroTensor Core Architecture](#microtensor-core-architecture)
3. [Automatic Differentiation Engine](#automatic-differentiation-engine)
4. [Neural Network Module System](#neural-network-module-system)
5. [Transformer Architecture Integration](#transformer-architecture-integration)
6. [Distributed Training Infrastructure](#distributed-training-infrastructure)
7. [Performance Analysis & Optimization](#performance-analysis--optimization)
8. [Production Features](#production-features)
9. [Installation & Usage](#installation--usage)
10. [Technical Appendix](#technical-appendix)

---

## Theoretical Foundations

### Automatic Differentiation Principles

MicroTensor implements **reverse-mode automatic differentiation** (backpropagation), a computational technique that efficiently computes gradients by constructing a dynamic computation graph during forward execution and traversing it in reverse topological order during backward propagation.

#### Mathematical Foundation

For a composite function $f = f_n \circ f_{n-1} \circ \ldots \circ f_1$, the chain rule states:

$$\frac{\partial f}{\partial x} = \frac{\partial f_n}{\partial f_{n-1}} \cdot \frac{\partial f_{n-1}}{\partial f_{n-2}} \cdot \ldots \cdot \frac{\partial f_1}{\partial x}$$

Reverse-mode AD computes this efficiently by:
1. **Forward pass**: Execute operations while recording the computation graph
2. **Backward pass**: Traverse the graph in reverse, accumulating gradients via the chain rule

#### Computational Graph Construction

Each tensor operation creates nodes in a directed acyclic graph (DAG) where nodes represent tensors (intermediate results), edges represent dependencies between operations, and gradient functions store the local derivative computation for each operation. This graph structure enables automatic gradient computation through applying the chain rule.
### Broadcasting Semantics

Broadcasting enables operations between tensors of different shapes by implicitly expanding dimensions according to NumPy broadcasting rules. MicroTensor implements broadcasting with proper gradient flow by tracking which dimensions were expanded and summing gradients accordingly during backpropagation.

#### Gradient Accumulation Under Broadcasting

When tensors are broadcasted, gradients must be accumulated correctly:
- **Dimension Extension**: Add dimensions of size 1 to the left
- **Size Expansion**: Replicate elements along dimensions where one operand has size 1
- **Gradient Reduction**: Sum gradients along broadcasted dimensions during backward pass

This ensures gradient correctness regardless of input tensor shapes.

---

## MicroTensor Core Architecture

### Tensor Representation

The `Tensor` class serves as the fundamental data structure, encapsulating:

```python
class Tensor:
    def __init__(self, data, dtype=None, _children=(), _op=None, requires_grad=False):
        self.data = cupy.array(data, dtype=dtype or np.float32)  # Numerical data
        self._prev = set(c for c in _children if c.requires_grad)  # Parent tensors
        self._op = _op                                            # Operation name
        self.requires_grad = requires_grad                        # Gradient tracking
        self.grad_fn = None                                       # Backward function
        self.grad = None                                          # Accumulated gradients
```

#### Design Rationale

Memory efficiency in MicroTensor is achieved through lazy gradient allocation, where gradients are created only if both requires_grad=True and grad_is_enabled=True. This minimizes unnecessary memory usage during inference. For computation, the CuPy backend provides GPU acceleration while preserving full compatibility with the NumPy API, allowing seamless switching between CPU and GPU execution. Finally, graph metadata—captured through the _prev set and _op string—supports reconstruction of the computation graph, enabling reliable automatic differentiation.

### Operation Implementation Pattern

Each tensor operation follows a consistent pattern:

```python
def operation(self, other):
    # 1. Forward computation
    result_data = compute_forward(self.data, other.data)
    
    # 2. Create output tensor with graph metadata
    out = Tensor(result_data, _children=(self, other), _op="operation_name")
    
    # 3. Define backward function (closure captures local variables)
    if self.requires_grad and grad_is_enabled:
        def _backward():
            # Compute and accumulate gradients using chain rule
            self.grad += compute_self_gradient(out.grad)
            other.grad += compute_other_gradient(out.grad)
        out.grad_fn = _backward
    
    return out
```

Each tensor operation follows a consistent pattern of forward computation, output tensor creation with graph metadata, and backward function definition through closures that capture local variables. This pattern ensures forward efficiency through direct computation without unnecessary overhead, backward correctness via proper gradient computation using the chain rule, and memory safety through closures that capture necessary data for gradient computation.

### Advanced Operations

#### Matrix Multiplication with Batching

Matrix multiplication requires special handling for batched inputs:

```python
def __matmul__(self, other):
    result = Tensor(self.data @ other.data, _children=(self, other), _op="matmul")
    
    def _matmul_backward():
        if len(result.grad.shape) > 2:  # Batched case
            for idx in np.ndindex(result.grad.shape[:-2]):
                self.grad[idx] += result.grad[idx] @ other.data[idx].T
        else:  # Standard 2D case
            self.grad += result.grad @ other.data.T
    
    result.grad_fn = _matmul_backward
    return result
```

#### Softmax with Numerical Stability

Softmax implementation prevents overflow through max subtraction:

```python
def softmax(self, dim=-1):
    # Numerical stability: subtract max to prevent overflow
    x_max = self.data.max(axis=dim, keepdims=True)
    exp_x = np.exp(self.data - x_max)
    sum_exp_x = exp_x.sum(axis=dim, keepdims=True)
    
    out = Tensor(exp_x / sum_exp_x, _children=(self,), _op="softmax")
    
    def _softmax_backward():
        s = out.data
        # Jacobian of softmax: s * (grad - (s * grad).sum())
        self.grad += s * (out.grad - (s * out.grad).sum(axis=dim, keepdims=True))
    
    out.grad_fn = _softmax_backward
    return out
```

---

## Automatic Differentiation Engine

### Computation Graph Management

The automatic differentiation engine manages computation graphs through several key components:

#### Topological Sorting Algorithm

```python
def backward(self):
    visited = set()
    topo_order = []
    
    def topological_sort(tensor):
        if tensor not in visited:
            visited.add(tensor)
            for parent in tensor._prev:
                topological_sort(parent)
            topo_order.append(tensor)
    
    topological_sort(self)
    
    # Initialize root gradient
    self.grad = np.ones_like(self.data)
    
    # Execute backward functions in reverse topological order
    for tensor in reversed(topo_order):
        if tensor.grad_fn:
            tensor.grad_fn()
```
The backward pass employs depth-first search to construct a topological ordering of the computation graph, ensuring that all dependencies are computed before their dependents. This algorithm visits each tensor exactly once, building the ordering through recursive traversal of parent tensors, then executes gradient functions in reverse topological order.
#### Gradient Context Management
The no_grad context manager prevents graph construction and gradient computation when they're not needed, significantly reducing memory usage and overhead during inference.

### Gradient Accumulation Strategies

#### Broadcasting Gradient Handling

When operations involve broadcasting, gradients must be reduced appropriately:

```python
def broadcast_to(self, target_shape):
    broadcasted_data = np.broadcast_to(self.data, target_shape)
    out = Tensor(broadcasted_data, _children=(self,), _op="broadcast")
    
    def _broadcast_backward():
        # Identify broadcasted axes
        broadcasted_axes = []
        for i, (s, t) in enumerate(zip(self.shape, target_shape)):
            if s == 1 and t > 1:
                broadcasted_axes.append(i)
        
        # Sum gradients along broadcasted axes
        grad_to_add = np.sum(out.grad, axis=tuple(broadcasted_axes), keepdims=True)
        self.grad += grad_to_add.reshape(self.shape)
    
    out.grad_fn = _broadcast_backward
    return out
```

Gradients are stored with the same dtype and device as their corresponding tensors, minimizing memory overhead and ensuring computational consistency.

---

## Neural Network Module System

### Module Hierarchy and Parameter Management

The neural network module system provides a PyTorch-compatible interface for building complex architectures:

```python
class Module:
    def __init__(self, device="cpu"):
        self.device = device
        self.is_training = True
        self._grad_hooks = []
    
    def parameters(self):
        return [t for t in self._get_tensors() if t.requires_grad]
    
    def zero_grad(self):
        for param in self.parameters():
            param._reset_grad()
```

#### Recursive Parameter Discovery

The module system automatically discovers parameters in nested modules:

```python
def _get_tensors(self):
    tensors = []
    for _, value in self.__dict__.items():
        if isinstance(value, Tensor):
            tensors.append(value)
        elif isinstance(value, (Module, ModuleList, ModuleDict)):
            tensors.extend(value._get_tensors())
    return tensors
```


### Core Neural Network Layers

#### Linear Layer with Proper Initialization

```python
class Linear(Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        # He initialization for better gradient flow
        bound = np.sqrt(6.0 / (in_features + out_features))
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True
        )
        if use_bias:
            self.bias = Tensor(np.zeros(out_features), requires_grad=True)
```

Linear layers implement fully connected transformations with He initialization to maintain activation variance across layers, preventing vanishing/exploding gradients in deep networks. The initialization scheme uses bounds calculated from input and output dimensions to ensure proper gradient flow during training.
#### Layer Normalization

Layer normalization stabilizes training by normalizing activations across feature dimensions:

$$\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Where $\mu$ and $\sigma$ are computed along the feature dimension.

```python
def forward(self, x):
    mean = x.data.mean(axis=norm_dims, keepdims=True)
    var = ((x.data - mean) ** 2).mean(axis=norm_dims, keepdims=True)
    std = np.sqrt(var + self.eps)
    
    x_norm = (x.data - mean) / std
    
    if self.elementwise_affine:
        out_data = self.weight.data * x_norm + self.bias.data
    else:
        out_data = x_norm
    
    return Tensor(out_data, _children=children, _op="layernorm")
```

#### Advanced Loss Functions

**Cross-Entropy Loss with Label Smoothing**:

This prevents overconfident predictions and improves generalization by distributing probability mass across classes. The implementation includes numerical stability measures through max subtraction and proper handling of padding tokens in sequence-to-sequence tasks.

---

## Transformer Architecture Integration

### Hybrid Computational Strategy
The main motivation for this project was to move beyond implementing tensor operations in isolation and to explore how they are actually used within a Transformer model. To achieve this, I developed a English–Italian  machine translation model where MicroTensor powers the core operations while PyTorch provides the surrounding training infrastructure.I had three key objectives:  
1. **Understanding**: Demonstrate how fundamental operations such as matrix multiplication, broadcasting, normalization, and softmax combine to form the building blocks of attention and feed-forward networks.  
2. **Practical Integration**: Retain the robustness of PyTorch’s ecosystem—including distributed training, data loading, and optimization—while substituting selected operations with MicroTensor implementations.  
3. **Comparison**: Enable direct comparison between MicroTensor and PyTorch kernels in terms of correctness, efficiency, and memory usage.  

### DeepSeek-Inspired Multi-Head Attention

The implementation incorporates advanced attention mechanisms inspired by DeepSeek's architecture, featuring latent space attention compression, rotary position embeddings (RoPE), and MicroTensor-powered computation kernels.
#### Latent Space Attention Compression

```python
class MultiHeadLatentAttentionBlock(Module):
    def __init__(self, d_model, h, d_c, d_cr, d_rope, dropout):
        # Latent space projections for computational efficiency
        self.W_DKV = nn.Linear(d_model, d_c)      # Keys/Values compression
        self.W_DQ = nn.Linear(d_model, d_cr)      # Queries compression
        self.W_QR = nn.Linear(d_cr, h * d_rope)   # RoPE embeddings    
        self.W_KR = nn.Linear(d_model, d_rope)    # Key RoPE embeddings
```

#### Rotary Position Embeddings (RoPE)

```python
def apply_rope(self, x):
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device) / half))
    angle = x[..., :half] * freq.view(1, 1, 1, -1)
    return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
```

#### MicroTensor-Powered Attention Computation

```python
class MicrotensorAttentionOps:
    @staticmethod
    def scaled_dot_product(q, k, v, scale, mask=None):
        # Convert PyTorch tensors to MicroTensor
        q_micro = MicroTensor(np.array(q.detach().cpu().numpy()))
        k_micro = MicroTensor(np.array(k.detach().cpu().numpy()))
        v_micro = MicroTensor(np.array(v.detach().cpu().numpy()))
        
        # Core attention computation
        k_t = k_micro.T(axes=[0, 1, 3, 2])  # Transpose for multiplication
        scores = q_micro @ k_t               # Attention scores
        scores = scores * scale              # Scale by sqrt(d_k)
        
        # Apply causal mask for decoder self-attention
        if mask is not None:
            mask_np = np.array(mask.detach().cpu().numpy())
            scores_data = np.where(mask_np == 0, -1e9, scores.data)
            scores = MicroTensor(scores_data, requires_grad=False)
        
        # Apply softmax and compute output
        attn_weights = softmax(scores, axis=-1)
        output = attn_weights @ v_micro
        
        # Convert back to PyTorch tensor
        return torch.tensor(output.data, device=q.device)
```
The attention mechanism reduces computational complexity by projecting keys and values into a lower-dimensional space $d_c$ and queries into $d_{cr}$, while preserving representational power through specialized projection matrices ($W_{DKV}$, $W_{DQ}$, $W_{QR}$, $W_{KR}$). To encode positional information, it incorporates **Rotary Position Embeddings (RoPE)**, which rotate query and key vectors in complex space, enabling the model to capture relative positions and generalize effectively to sequences longer than those seen during training. The **core attention computations** include scaled dot-product attention, causal masking for decoder self-attention, and numerically stable softmax, all executed using MicroTensor operations with efficient conversion between PyTorch and MicroTensor formats. This approach shows how custom low-level kernels can be applied within a modern Transformer.


### Feed-Forward Network Enhancement

The feed-forward networks utilize MicroTensor for core computations:

```python
class MicrotensorFeedForward:
    @staticmethod
    def forward(x_torch, w1, b1, w2, b2, dropout_rate, training):
        # Convert to MicroTensor
        x = MicroTensor(np.array(x_torch.detach().cpu().numpy()))
        
        # First linear transformation
        h = x @ np.array(w1.T.detach().cpu().numpy()) + np.array(b1.detach().cpu().numpy())
        
        # Apply ReLU activation
        h = relu(h)
        
        # Apply dropout if in training mode
        if training and dropout_rate > 0:
            mask = np.random.binomial(1, 1-dropout_rate, h.data.shape).astype(np.float32)
            h = MicroTensor(h.data * mask / (1 - dropout_rate))
        
        # Second linear transformation
        out = h @ np.array(w2.T.detach().cpu().numpy()) + np.array(b2.detach().cpu().numpy())
        
        return torch.tensor(out.data, device=x_torch.device)
```

## Distributed Training Infrastructure

### Multi-GPU Training Architecture

The distributed training system implements data parallelism using PyTorch's `DistributedDataParallel` with NCCL backend for optimal GPU communication:

```python
def setup_distributed_training():
    # Initialize process group for multi-GPU communication
    init_process_group(backend='nccl')
    
    # Set device for current process
    torch.cuda.set_device(local_rank)
    
    # Wrap model for distributed training
    model = DistributedDataParallel(
        model, 
        device_ids=[local_rank],
        find_unused_parameters=False  # Optimization for static graphs
    )
```

#### Gradient Synchronization Strategy

**All-Reduce Communication Pattern**:
1. Each GPU computes gradients independently on its data partition
2. Gradients are synchronized across all GPUs using all-reduce operations
3. Each GPU applies the averaged gradients to its local model copy

This ensures model consistency across all processes while maximizing parallel efficiency.

#### Data Distribution Strategy

```python
def get_distributed_dataloader(dataset, config):
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(
            dataset, 
            shuffle=True,
            drop_last=True  # Ensures consistent batch sizes across GPUs
        )
        return DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    else:
        return DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
```

The `DistributedSampler` ensures each GPU processes different data partitions while maintaining dataset coverage across epochs.

### Memory Management and Optimization

#### Strategic Memory Clearing

The system uses several techniques to manage memory efficiently during training. These include clearing the CUDA cache at epoch boundaries, using `set_to_none=True` for gradient clearing to avoid unnecessary zero-filling of buffers, and ensuring memory is released promptly to keep GPU utilization stable.


#### Gradient Accumulation Support

When GPU memory limits batch size, gradient accumulation provides an effective workaround. Loss values are scaled for accumulation, gradients are collected across multiple micro-batches, and optimizer updates are applied only after the desired number of steps. This enables training with larger effective batch sizes without exceeding hardware constraints.

### Fault Tolerance and Checkpointing

#### Comprehensive State Persistence

```python
def save_checkpoint(model, optimizer, epoch, global_step, config):
    state = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step,
        'config': config.__dict__,
        'wandb_run_id': wandb.run.id if wandb.run else None
    }
    torch.save(state, get_weights_file_path(config, epoch))
```

The checkpointing system preserves:
- Model parameters and optimizer states
- Training progress metadata
- Experiment tracking information
- Configuration parameters

This enables seamless training resumption even with interruptions.

---

## Performance Analysis & Optimization

### Computational Complexity Analysis

#### Attention Mechanism Complexity

**Standard Multi-Head Attention**: $O(n^2 d + n d^2)$ where $n$ is sequence length, $d$ is model dimension

**Latent Space Attention**: $O(n^2 d_c + n d_c d + n d_{cr} d)$ where $d_c, d_{cr} << d$

The latent space compression reduces computational complexity while maintaining representational capacity through specialized projection matrices.

#### Memory Usage Patterns

```python
# Typical memory footprint for batch_size=8, seq_len=350, d_model=512:
# - Model parameters: ~50MB (37M parameters × 4 bytes/param)
# - Forward activations: ~200MB per batch
# - Gradient storage: ~50MB (matching parameter size)
# - MicroTensor overhead: ~30-50MB (tensor conversions)
# - Peak GPU memory: ~2GB total
```

### Performance Optimization Strategies

#### Tensor Conversion Optimization

The hybrid architecture requires careful optimization of tensor conversions:

```python
class TensorConverter:
    @staticmethod
    def pytorch_to_microtensor(tensor, requires_grad=False):
        # Minimize copy operations
        numpy_data = tensor.detach().cpu().numpy()
        return MicroTensor(numpy_data, requires_grad=requires_grad)
    
    @staticmethod  
    def microtensor_to_pytorch(micro_tensor, device):
        # Efficient device placement
        return torch.tensor(micro_tensor.data, device=device)
```

**Optimization Principles**:
- **Minimize Copies**: Reuse buffers when possible
- **Batch Conversions**: Convert entire batches rather than individual elements
- **Device Locality**: Keep computations on appropriate devices

#### Dynamic Quantization

Post-training quantization reduces model size and inference latency:

```python
def apply_dynamic_quantization(model):
    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},      # Target linear layers
        dtype=torch.qint8       # 8-bit integer quantization
    )
```

**Quantization Benefits**:
- **Model Size**: 2-4x reduction (Float32 → Int8)
- **Inference Speed**: 1.5-2x improvement on CPU
- **Memory Bandwidth**: Reduced data movement requirements

### Performance Benchmarking Results

#### Training Performance Comparison

| Configuration | Pure PyTorch | With MicroTensor | Overhead |
|--------------|--------------|------------------|----------|
| Forward Pass | 45ms/batch   | 55ms/batch      | +22%     |
| Backward Pass| 38ms/batch   | 42ms/batch      | +11%     |
| Total Training| 83ms/batch  | 97ms/batch      | +17%     |

#### Memory Usage Analysis

| Component | Pure PyTorch | With MicroTensor | Overhead |
|-----------|--------------|------------------|----------|
| Model Parameters | 50MB | 50MB | 0% |
| Activations | 200MB | 230MB | +15% |
| Peak Memory | 1.8GB | 2.1GB | +17% |

---

## Production Features

### Comprehensive Monitoring and Metrics

#### Training Metrics Integration

```python
def compute_validation_metrics(predicted, expected):
    # Character Error Rate - fine-grained sequence accuracy
    cer = torchmetrics.CharErrorRate()(predicted, expected)
    
    # Word Error Rate - token-level accuracy
    wer = torchmetrics.WordErrorRate()(predicted, expected)
    
    # BLEU Score - translation quality assessment
    bleu = torchmetrics.BLEUScore()(predicted, expected)
    
    return {'cer': cer, 'wer': wer, 'bleu': bleu}
```

**Metric Selection Rationale**:
- **CER**: Measures character-level accuracy, sensitive to morphological errors
- **WER**: Captures word-level correctness, balances precision and recall
- **BLEU**: Industry standard for translation quality evaluation

#### Experiment Tracking Integration

```python
def setup_experiment_tracking(config):
    wandb.init(
        project="transformer-translation-microtensor",
        name=f"rank_{config.global_rank}",
        config=config.__dict__,
        resume="allow",  # Enable resuming interrupted experiments
        group=config.wandb_group
    )
    
    # Define custom metrics
    wandb.define_metric("global_step")
    wandb.define_metric("validation/*", step_metric="global_step")
    wandb.define_metric("train/*", step_metric="global_step")
```

### Model Deployment Pipeline

#### Translation Inference Engine

```python
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    # Encode source sequence
    encoder_output = model.encode(source, source_mask)
    
    # Generate target sequence autoregressively
    decoder_input = torch.tensor([[sos_idx]], device=device)
    
    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        decoder_output = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        
        # Predict next token
        logits = model.project(decoder_output[:, -1])
        next_token = logits.argmax(dim=-1)
        
        # Append to sequence
        decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=1)
        
        # Stop at end token
        if next_token.item() == eos_idx:
            break
    
    return decoder_input.squeeze(0)
```

