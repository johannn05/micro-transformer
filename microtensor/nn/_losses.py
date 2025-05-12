from ..core import Tensor, DEFAULT_MIN
from ._module import Module
import cupy as np

def _check_tensor_types(a: Tensor, b: Tensor):
    """
    Check if both tensors are compatible for operations.

    Args:
        a (Tensor): First tensor.
        b (Tensor): Second tensor.

    Raises:
        RuntimeError: If the tensors are incompatible.
    """
    if type(a.data) != type(b.data):
        raise RuntimeError("Expected both Tensors to be of the same type.")

class CrossEntropyLoss(Module):
    """
    Computes cross entropy loss between input logits and target classes.
    Essential for training transformer models on language tasks.
    """
    def __init__(
        self,
        ignore_index: int = -100,  # Index to ignore (usually padding)
        label_smoothing: float = 0.0,  # Amount of smoothing to apply
        reduction: str = "mean"  # How to reduce the loss
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError(f"Label smoothing must be in [0,1], got {label_smoothing}")

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute cross entropy loss between predictions and targets.
        """
        _check_tensor_types(pred, target)
        
        # Create mask for non-padding tokens
        target_flat = target.reshape(-1)
        valid_mask = np.not_equal(target_flat.data, self.ignore_index)
        num_valid = np.sum(valid_mask)
        
        # Compute log softmax with numerical stability
        pred_flat = pred.reshape(-1, pred.shape[-1])
        valid_preds = pred_flat.data[valid_mask]
        valid_targets = target_flat.data[valid_mask]
        
        # Compute softmax with numerical stability
        max_logits = np.max(valid_preds, axis=-1, keepdims=True)
        exp_logits = np.exp(valid_preds - max_logits)
        softmax_output = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        log_softmax = np.log(softmax_output + 1e-8)  # Add small epsilon for numerical stability
        
        # Create target distribution with label smoothing
        target_dist = np.zeros_like(valid_preds)
        target_dist[np.arange(len(valid_targets)), valid_targets] = 1
        if self.label_smoothing > 0.0:
            target_dist = (1 - self.label_smoothing) * target_dist + self.label_smoothing / pred.shape[-1]
        
        # Compute loss
        loss_value = -np.sum(target_dist * log_softmax) / (num_valid if self.reduction == "mean" else 1)
        
        # Create output tensor with gradient tracking
        out = Tensor(
            loss_value,
            dtype=pred.dtype,
            _children=(pred,),
            _op="cross_entropy",
            requires_grad=True  # Always True for loss
        )
        
        # Set up gradient function
        if pred.requires_grad and Tensor.grad_is_enabled:
            def _cross_entropy_backward():
                if out.grad is None:
                    return
                    
                # Initialize gradient
                grad = np.zeros_like(pred_flat.data)
                
                # Compute gradient
                grad_valid = (softmax_output - target_dist)
                if self.reduction == "mean":
                    grad_valid /= num_valid
                
                # Scale by output gradient
                grad_valid *= out.grad
                
                # Place gradients back in correct positions
                grad[valid_mask] = grad_valid
                
                # Reshape gradient to match input
                pred.grad = pred.grad + grad.reshape(pred.shape) if pred.grad is not None else grad.reshape(pred.shape)
            
            out.grad_fn = _cross_entropy_backward
        
        return out
        
class MSELoss(Module):
    """
    creates a criterion that measures the mean squared error (squared l2 norm) 
    between each element in pred and target of size n based on reduction.

    if reduction is "sum", it doesn't divide by n to get mean.
    if reduction is "mean", it divides by n to get mean.
    """
    def __init__(self, reduction: str = "sum") -> None:
        super().__init__()  # ensure proper superclass initialization
        self.reduction = reduction

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        """
        computes the mean squared error loss between predictions and targets.
        """
        _check_tensor_types(pred, actual)

        l2sum = ((pred - actual) ** 2).sum()

        if self.reduction == "sum":
            return l2sum
        elif self.reduction == "mean":
            return l2sum / actual.shape[0]
        else:
            raise ValueError(f"invalid reduction type '{self.reduction}' found.")


class BCELoss(Module):
    """
    calculates binary cross-entropy loss between predictions and targets.
    """
    def __init__(self, eps: float = DEFAULT_MIN) -> None:
        super().__init__()  # ensure proper superclass initialization
        self.eps = eps

    def forward(self, pred: Tensor, actual: Tensor) -> Tensor:
        """
        computes binary cross-entropy loss with numerical stability.
        """
        _check_tensor_types(pred, actual)

        # clip the predictions and targets to avoid numerical instability
        a: Tensor = pred * actual.clip(self.eps, 1 - self.eps).log()
        b: Tensor = (1 - pred) * (1 - actual).clip(self.eps, 1 - self.eps).log()

        return -(a + b).sum() / pred.shape[0]
