import cupy as np
from typing import *
from functools import partial  


# constants
DEFAULT_MIN = 1e-6
DEFAULT_MAX = 1 - 1e-6

Array = np.ndarray

def _get_d(device: str):
    """
    Returns the appropriate device-specific backend or library.
    
    Args:
        device (str): The device name ("cpu" or "gpu").
        
    Returns:
        module: The backend library for the specified device.
    """
    import cupy as np #for gpu support
    return np

class no_grad:
    """
    Context-manager that disables gradient calculation.
    """

    def __enter__(self):
        self.previous = Tensor.grad_is_enabled
        Tensor.grad_is_enabled = False

    def __exit__(self, exc_type, exc_value, traceback):
        Tensor.grad_is_enabled = self.previous


class Tensor:
    """
    A simplified Tensor class using numpy as the backend.
    """
    grad_is_enabled: bool = True

    def __init__(
        self,
        data: Union[np.ndarray, Any],
        dtype=None,
        _children: tuple = (),
        _op=None,
        requires_grad: bool = False
    ) -> None:
        """
        Initialize a tensor.

        Args:
            data: The input data (array-like or numpy.ndarray).
            dtype: The data type of the tensor elements (default: np.float32).
            _children: Parent tensors in the computation graph (default: ()).
            _op: The operation that created this tensor (default: None).
            requires_grad: Whether the tensor should track gradients.
        """
        self.dtype = dtype or np.float32

        # convert input data to numpy array
        self.data = (
            np.array(data, dtype=self.dtype)
            if not isinstance(data, np.ndarray)
            else data.astype(dtype=self.dtype)
        )

        # Parent tensors and operation metadata
        self._prev = set([c for c in _children if c.requires_grad])
        self._op = _op

        # Gradient-related attributes
        self.requires_grad = requires_grad
        self.grad = (
            np.zeros_like(self.data, dtype=self.dtype) if self.requires_grad and self.grad_is_enabled else None
        )
        self.grad_fn = None  # Function to compute gradients for this tensor

        # Tensor properties
        self.shape = self.data.shape
        self.ndim = len(self.shape)

        # Hooks to run after calculating the gradient
        self._grad_hooks: List[Callable] = []

    def _reset_grad(self) -> None:
        """
        Reset the gradient to zero.
        """
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data, dtype=self.dtype)
            else:
                self.grad.fill(0)


    def set_requires_grad(self, val: bool) -> None:
        """
        Set whether the tensor requires gradients.

        Args:
            val (bool): True to enable gradient tracking; False to disable it.
        """
        if not isinstance(val, bool):
            raise ValueError("The value for requires_grad must be a boolean.")

        if val and self.grad is None:  # If gradients are enabled but not initialized
            self._reset_grad()

        self.requires_grad = val

    def register_grad_hook(self, hook: Callable) -> None:
        """
        Add a hook to be called after the calculation of gradients.
        """
        assert hook not in self._grad_hooks
        self._grad_hooks.append(hook)

    def reset_grad_hooks(self) -> None:
        """
        Reset all gradient hooks.
        """
        self._grad_hooks = []

    def backward(self) -> None:
        """
        Perform backpropagation to compute gradients for the computation graph.

        This method computes gradients for all tensors in the computation graph
        starting from the current tensor. It traverses the graph in reverse topological
        order and applies the gradient functions (_grad_fn) for each tensor.

        Raises:
            ValueError: If gradients are disabled for the computation graph.
        """
        if not self.grad_is_enabled:
            raise ValueError("Cannot perform backward when gradient calculation is disabled.")

        if self.grad is None:
            raise ValueError("This tensor does not require gradients.")

        # Topological sort of the computation graph
        visited = set()
        topo_order = []

        def _topological_sort(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for parent in tensor._prev:
                    _topological_sort(parent)
                topo_order.append(tensor)

        _topological_sort(self)

        # Gradient of the current tensor with respect to itself is always 1
        self.grad = np.ones_like(self.data, dtype=self.dtype)

        # Initialize gradients for all tensors in the graph
        for tensor in topo_order:
            if tensor.requires_grad and tensor.grad is None:
                tensor.grad = np.zeros_like(tensor.data, dtype=tensor.dtype)

        # Traverse the graph in reverse topological order
        for tensor in reversed(topo_order):
            if tensor.grad_fn is not None:
                tensor.grad_fn()  # Apply the gradient function
            # Execute gradient hooks if any
            for hook in tensor._grad_hooks:
                hook(tensor)

    def clip(
        self,
        min_value: float = DEFAULT_MIN, # min value for clipping tensor's data.
        max_value: float = DEFAULT_MAX, # min value for clipping tensor's data.
        clip_gradients: bool = False, # whether to clip the gradients as well.
        grad_min_value: float = DEFAULT_MIN, #min for clippling grads
        grad_max_value: float = DEFAULT_MAX, #max for clippling grads
    ) -> "Tensor":
        """
        Clip the tensor's data and optionally its gradients to specified ranges.
        """
        # Clip data to the specified range
        self.data = np.clip(self.data, min_value, max_value)

        # optionally, clip gradients to the specified range
        if clip_gradients and self.grad is not None:
            self.grad = np.clip(self.grad, grad_min_value, grad_max_value)

        return self

    
    def __setitem__(self, indices, value: "Tensor") -> None:
        """
        Set values of the tensor at specified indices with another tensor's values.
        """
        if not isinstance(value, Tensor):
            raise ValueError("The value must be a Tensor.")

        # update tensor's data
        self.data[indices] = np.array(value.data, dtype=self.dtype)

        # update tensor's gradients if gradients are tracked
        if self.requires_grad and value.requires_grad:
            if self.grad is None or value.grad is None:
                raise ValueError("Cannot set gradients: either self or value has no gradient.")
            self.grad[indices] = np.array(value.grad, dtype=self.dtype)

    def __getitem__(self, indices):
        """
        Get a subset of the tensor using indices.
        """
        out = Tensor(
            self.data[indices],  # Sliced data
            dtype=self.dtype,
            _children=(self,),
            _op="getitem",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def _getitem_backward():
                grad = np.zeros_like(self.data, dtype=self.dtype)
                grad[indices] = out.grad
                self.grad += grad

            out.grad_fn = _getitem_backward

        return out


    # ----------------------- UNARY OPS --------------------------------
    def broadcast_to(self, target_shape: Tuple[int]) -> "Tensor":
        """
        broadcast tensor to a specified shape, which allows a tensor to expand its dimensions 
        or replicate elements along certain axes to match the target shape.

        Args:
            target_shape (Tuple[int]): The shape to broadcast the tensor to.

        Returns:
            Tensor: A new tensor with the broadcasted shape.
        """
        # broadcast data to the target shape
        broadcasted_data = np.broadcast_to(self.data, target_shape)

        # create a new tensor with the broadcasted data
        out = Tensor(
            broadcasted_data,
            dtype=self.dtype,
            _children=(self,),
            _op="broadcast",
            requires_grad=self.requires_grad
        )

        # if gradients enabled, set up  backward function
        if self.requires_grad and self.grad_is_enabled:
            # determine axes that were broadcasted
            broadcasted_axes = [
                i for i, (s, t) in enumerate(zip((1,) * (len(target_shape) - len(self.shape)) + self.shape, target_shape))
                if s == 1 and t > 1
            ]

            def _broadcast_backward():
                # sum gradients along broadcasted axes
                grad_to_add = np.sum(out.grad, axis=tuple(broadcasted_axes), keepdims=True)
                # squeeze to match the original shape of self.grad
                grad_to_add = np.reshape(grad_to_add, self.shape)
                self.grad += grad_to_add

            out.grad_fn = _broadcast_backward

        return out






    def sum(self, axis: Union[int, Tuple[int]] = None, keepdims: bool = False) -> "Tensor":
        """
        Sum values of the tensor along specified axes.

        Args:
            axis (int or Tuple[int], optional): The axes along which to sum.
            keepdims (bool, optional): Whether to retain reduced dimensions.

        Returns:
            Tensor: A new tensor with summed values.
        """
        summed_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            summed_data,
            dtype=self.dtype,
            _children=(self,),
            _op="sum",
            requires_grad=self.requires_grad  # Propagate requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def _sum_backward():
                grad = np.ones_like(self.data) if axis is None else np.expand_dims(out.grad, axis=axis)
                self.grad += grad if keepdims else grad.reshape(self.data.shape)

            out.grad_fn = _sum_backward

        return out


    
    def mean(self, axis: Union[int, Tuple[int]] = None, keepdims: bool = False) -> "Tensor":
        """
        calculate the mean of the tensor elements along specified axes.
        Returns:
            Tensor: A new tensor with mean values.
        """
        N = np.prod(self.data.shape if axis is None else np.array(self.data.shape)[axis])
        out = self.sum(axis=axis, keepdims=keepdims) / N
        return out

    
    def _stdvar_helper__(self, axis: int = None, keepdims: bool = False, correction: int = 0) -> "Tensor":
        """
        helper function for variance and standard deviation calculations.
        Returns:
            Tensor: A tensor representing variance.
        """
        N = self.data.shape[axis] if axis is not None else self.data.size
        if N - correction <= 0:
            raise ValueError("Degrees of freedom must be less than the number of elements.")
        squared_diff = (self - self.mean(axis=axis, keepdims=True)) ** 2
        return squared_diff.sum(axis=axis, keepdims=keepdims) / (N - correction)

    
    def std(self, axis: int = None, keepdims: bool = False, correction: int = 0) -> "Tensor":
        """
        calculate the standard deviation of the Tensor elements along given axis
        """
        t1 = self._stdvar_helper__(axis=axis, keepdims=keepdims, correction=correction)
        return t1 ** (1/2)
    
    def var(self, axis: int = None, keepdims: bool = False, correction: int = 0) -> "Tensor":
        """
        calculate variance along given axis and correction
        """
        return self._stdvar_helper__(axis=axis, keepdims=keepdims, correction=correction)
    
    def half(self) -> "Tensor":
        """
        convert the tensor's data and gradients to half precision (float16).
        Returns:
            Tensor: A new tensor with half-precision data.
        """
        if self.dtype != np.float32:
            raise ValueError(f"Tensor must be float32, got {self.dtype}")
        
        half_data = self.data.astype(np.float16)
        out = Tensor(half_data, dtype=np.float16, _children=(self,), _op="half")

        if self.requires_grad and self.grad_is_enabled:
            def _half_backward():
                self.grad += out.grad.astype(np.float32)

            out.grad_fn = _half_backward

        return out


    def T(self, axes: Optional[Iterable[int]] = None) -> "Tensor":
        """Transpose the tensor along specified axes."""
        transposed_data = np.transpose(self.data, axes=axes)
        out = Tensor(
            transposed_data,
            dtype=self.dtype,
            _children=(self,),
            _op="transpose",
            requires_grad=self.requires_grad  # Maintain requires_grad
        )

        if self.requires_grad and Tensor.grad_is_enabled:
            def _transpose_backward():
                if out.grad is not None:
                    self.grad = (self.grad if self.grad is not None else 0) + np.transpose(out.grad, axes=None if axes is None else np.argsort(axes))

            out.grad_fn = _transpose_backward

        return out


    
    def exp(self) -> "Tensor":
        """
        compute  exponential of the tensor's elements.
        Returns:
            Tensor: A new tensor with exponential values.
        """
        exp_data = np.exp(self.data)
        out = Tensor(exp_data, dtype=self.dtype, _children=(self,), _op="exp")

        if self.requires_grad and self.grad_is_enabled:
            def _exp_backward():
                self.grad += exp_data * out.grad

            out.grad_fn = _exp_backward

        return out


    def log(self) -> "Tensor":
        """
        compute natural logarithm of the tensor's elements.

        Returns:
            Tensor: A new tensor with logarithm values.
        """
        log_data = np.log(self.data)
        out = Tensor(log_data, dtype=self.dtype, _children=(self,), _op="log")

        if self.requires_grad and self.grad_is_enabled:
            def _log_backward():
                self.grad += out.grad / self.data

            out.grad_fn = _log_backward

        return out

    
    def reshape(self, *shape: int) -> "Tensor":
        """Reshape the tensor to the specified shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
            
        reshaped_data = self.data.reshape(shape)
        out = Tensor(
            reshaped_data,
            dtype=self.dtype,
            _children=(self,),
            _op="reshape",
            requires_grad=self.requires_grad  # Maintain requires_grad
        )

        if self.requires_grad and Tensor.grad_is_enabled:
            def _reshape_backward():
                if out.grad is not None:
                    self.grad = (self.grad if self.grad is not None else 0) + out.grad.reshape(self.data.shape)

            out.grad_fn = _reshape_backward

        return out



    def masked_fill(self, mask: Union[np.ndarray, "Tensor"], value: Union[float, int]) -> "Tensor":
        """
        Enhanced masked_fill that properly handles float('-inf') for attention masks.
        
        Args:
            mask: Boolean mask where True indicates positions to fill
            value: Value to put in masked positions (often float('-inf') for attention)
            
        Returns:
            A new tensor with masked values filled
        """
        if isinstance(mask, Tensor):
            mask = mask.data
        
        # Handle float('-inf') explicitly to avoid numerical issues
        if value == float('-inf'):
            filled_data = np.where(mask, -np.inf, self.data)
        else:
            filled_data = np.where(mask, value, self.data)
        
        out = Tensor(
            filled_data,
            dtype=self.dtype,
            _children=(self,),
            _op="masked_fill",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def _masked_fill_backward():
                self.grad += np.where(mask, 0, out.grad)
            
            out.grad_fn = _masked_fill_backward

        return out


    # ------------------------ BINARY OPS -------------------------

    def cat(self, others: List["Tensor"], dim: Optional[int] = 0) -> "Tensor":
        """
        concatenate self and other tensors along given dimension
        """
        tocat: List[Tensor] = [self]
        for other in others:
            if not isinstance(other, Tensor):
                raise ValueError(f"Cannot concatenate type '{type(other)}'")
            tocat.append(other)

        concatenated_data = np.concatenate([t.data for t in tocat], axis=dim)
        out = Tensor(concatenated_data, dtype=self.dtype, _children=tuple(tocat), _op="cat")

        # grad handling
        if self.grad_is_enabled:
            out.set_requires_grad(True)
            sizes = [t.shape[dim] for t in tocat]
            splits = np.cumsum(sizes[:-1])
            grads = np.split(out.grad, splits, axis=dim)

            def _cat_backward():
                for t, grad in zip(tocat, grads):
                    if t.requires_grad:
                        t.grad += grad

            out.grad_fn = _cat_backward

        return out

    
    def split(self, sections: int, dim: int = 0) -> List["Tensor"]:
        """
        split tensor into equal sections along the given dimension.

        Returns:
            List[Tensor]: A list of tensors after splitting.
        """
        split_data = np.split(self.data, sections, axis=dim)
        outs = [Tensor(data, dtype=self.dtype, _children=(self,), _op="split") for data in split_data]

        # grad handling
        if self.requires_grad and self.grad_is_enabled:
            indices, start = [], 0
            for part in outs:
                idx = [slice(None)] * self.ndim
                idx[dim] = slice(start, start + part.shape[dim])
                indices.append(tuple(idx))
                start += part.shape[dim]

            def _split_backward(index: int = 0):
                self.grad[indices[index]] += outs[index].grad

            for i, part in enumerate(outs):
                part.grad_fn = partial(_split_backward, index=i)
                part.set_requires_grad(True)

        return outs


    def __matmul__(self, other: "Tensor") -> "Tensor":
        """Perform matrix multiplication."""
        if not isinstance(other, Tensor):
            other = Tensor(other)
        
        result = Tensor(
            self.data @ other.data,
            dtype=self.dtype,
            _children=(self, other),
            _op="matmul",
            requires_grad=self.requires_grad or other.requires_grad
        )
        
        if result.requires_grad and Tensor.grad_is_enabled:
            def _matmul_backward():
                if result.grad is None:
                    return
                    
                print("\nMatmul Backward Debug:")
                print(f"Result grad shape: {result.grad.shape}")
                print(f"Self data shape: {self.data.shape}")
                print(f"Other data shape: {other.data.shape}")
                
                # Handle multi-dimensional tensors
                if len(result.grad.shape) > 2:
                    if self.requires_grad:
                        # Initialize gradient if needed
                        if self.grad is None:
                            self.grad = np.zeros_like(self.data)
                        
                        # For each batch dimension, compute gradient
                        for idx in np.ndindex(result.grad.shape[:-2]):
                            self.grad[idx] += result.grad[idx] @ other.data[idx].T
                        
                    if other.requires_grad:
                        # Initialize gradient if needed
                        if other.grad is None:
                            other.grad = np.zeros_like(other.data)
                        
                        # For each batch dimension, compute gradient
                        for idx in np.ndindex(result.grad.shape[:-2]):
                            other.grad[idx] += self.data[idx].T @ result.grad[idx]
                else:
                    # Regular 2D matrix multiplication gradient
                    if self.requires_grad:
                        self.grad = np.zeros_like(self.data) if self.grad is None else self.grad
                        self.grad += result.grad @ other.data.T
                    
                    if other.requires_grad:
                        other.grad = np.zeros_like(other.data) if other.grad is None else other.grad
                        other.grad += self.data.T @ result.grad
            
            result.grad_fn = _matmul_backward
        
        return result


    def _preprocess_binop(self, other: Union["Tensor", Any]) -> Tuple["Tensor", "Tensor"]:
        """
        preprocess  two tensors for binary operations by broadcasting them.
        """
        if not isinstance(other, Tensor):
            other = Tensor(other)
        broadcast_shape = np.broadcast_shapes(self.shape, other.shape)
        return self.broadcast_to(broadcast_shape), other.broadcast_to(broadcast_shape)

    
    def __add__(self, other: Union["Tensor", Any]) -> "Tensor":
        """
        elementwise addition (with broadcasting).

        Args:
            other (Tensor or Any): The tensor or value to add.

        Returns:
            Tensor: A new tensor with the result of addition.
        """
        
        self, other = self._preprocess_binop(other)
        result_data = self.data + other.data
        out = Tensor(
            result_data,
            dtype=self.dtype,
            _children=(self, other),
            _op="add",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if self.grad_is_enabled and out.requires_grad:
            def _add_backward():
                if self.requires_grad:
                    self.grad += out.grad
                if other.requires_grad:
                    other.grad += out.grad

            out.grad_fn = _add_backward

        return out

    
    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        elementwise multiplication (supports broadcasting).
        """
        self, other = self._preprocess_binop(other)
        result_data = self.data * other.data
        out = Tensor(
            result_data,
            dtype=self.dtype,
            _children=(self, other),
            _op="mul",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if self.grad_is_enabled and out.requires_grad:
            def _mul_backward():
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad

            out.grad_fn = _mul_backward

        return out

    def __pow__(self, other: Union[int, float]) -> "Tensor":
        """
        raise tensor to a power (integer or float).
        """
        result_data = np.power(self.data, other)
        out = Tensor(
            result_data,
            dtype=self.dtype,
            _children=(self,),
            _op="pow",
            requires_grad=self.requires_grad,
        )

        if self.grad_is_enabled and out.requires_grad:
            def _pow_backward():
                self.grad += other * np.power(self.data, other - 1) * out.grad

            out.grad_fn = _pow_backward

        return out



    def __neg__(self) -> "Tensor":
        """
        Negate the tensor.
        """
        return self * -1


    def __sub__(self, other: "Tensor") -> "Tensor":
        """
        Elementwise subtraction.
        """
        return self + (-other)


    def __rsub__(self, other: "Tensor") -> "Tensor":
        """
        Reverse subtraction (other - self).
        """
        return -self + other


    def __radd__(self, other: "Tensor") -> "Tensor":
        """
        Reverse addition (other + self).
        """
        return self + other


    def __rmul__(self, other: "Tensor") -> "Tensor":
        """
        Reverse multiplication (other * self).
        """
        return self * other


    def __truediv__(self, other: Union["Tensor", Any]) -> "Tensor":
        self, other = self._preprocess_binop(other)
        result_data = self.data / other.data
        out = Tensor(
            result_data,
            dtype=self.dtype,
            _children=(self, other),
            _op="div",
            requires_grad=self.requires_grad or other.requires_grad,
        )

        if self.grad_is_enabled and out.requires_grad:
            def _div_backward():
                if self.requires_grad:
                    self.grad += (1 / other.data) * out.grad
                if other.requires_grad:
                    other.grad += (-self.data / (other.data**2)) * out.grad

            out.grad_fn = _div_backward

        return out



    def __rtruediv__(self, other: "Tensor") -> "Tensor":
        """
        Reverse division (other / self).
        """
        return (self ** -1) * other


    def __hash__(self) -> int:
        """
        Return the hash of the tensor (based on its ID).
        """
        return id(self)


    def __eq__(self, other: "Tensor") -> bool:
        """
        Check elementwise equality between two tensors.
        """
        if not isinstance(other, Tensor):
            return False
        return np.array_equal(self.data, other.data)


    def __repr__(self) -> str:
        """
        Return a string representation of the tensor.
        """
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"


    def view(self, *shape: int) -> "Tensor":
        """
        Functionally similar to reshape but maintains contiguous memory layout.
        Essential for attention operations in transformers.
        
        Args:
            shape: The new shape dimensions.
            
        Returns:
            A new tensor with the same data but different shape.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        reshaped_data = self.data.reshape(shape)
        out = Tensor(
            reshaped_data,
            dtype=self.dtype,
            _children=(self,),
            _op="view",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def _view_backward():
                self.grad += out.grad.reshape(self.data.shape)
            
            out.grad_fn = _view_backward

        return out

    def unsqueeze(self, dim: int) -> "Tensor":
        """
        Add a dimension of size 1 at the specified position.
        Used frequently in attention mechanisms for broadcasting.
        
        Args:
            dim: Position where new dimension should be inserted.
            
        Returns:
            A new tensor with an additional dimension.
        """
        new_shape = list(self.shape)
        new_shape.insert(dim if dim >= 0 else len(new_shape) + 1 + dim, 1)
        
        return self.reshape(new_shape)

    def softmax(self, dim: int = -1) -> "Tensor":
        """
        Compute softmax along specified dimension.
        Critical for attention score normalization.
        
        Args:
            dim: Dimension along which to compute softmax.
            
        Returns:
            A new tensor with softmax applied.
        """
        x_max = self.data.max(axis=dim, keepdims=True)
        exp_x = np.exp(self.data - x_max)
        sum_exp_x = exp_x.sum(axis=dim, keepdims=True)
        out = Tensor(
            exp_x / sum_exp_x,
            dtype=self.dtype,
            _children=(self,),
            _op="softmax",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and self.grad_is_enabled:
            def _softmax_backward():
                s = out.data
                self.grad += s * (out.grad - (s * out.grad).sum(axis=dim, keepdims=True))
            
            out.grad_fn = _softmax_backward

        return out
    
    def max(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """
        Computes the maximum value along the specified axis.
        
        When computing softmax or log-softmax, we often need to find the maximum 
        value to prevent numerical overflow. By subtracting the maximum before 
        taking the exponential, we can keep the values in a reasonable range.
        
        Args:
            axis: The axis along which to compute the maximum. If None, finds the
                global maximum of the entire tensor.
            keepdims: If True, the output tensor preserves the reduced dimensions
                    with size 1, making it easier to broadcast later.
        
        Returns:
            A new tensor containing the maximum values.
        """
        max_data = np.max(self.data, axis=axis, keepdims=keepdims)
        out = Tensor(
            max_data,
            dtype=self.dtype,
            _children=(self,),
            _op="max",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and Tensor.grad_is_enabled:
            def _max_backward():
                # Create a mask where data equals the maximum
                if axis is not None and not keepdims:
                    # Reshape max_data to broadcast correctly
                    max_reshaped = np.expand_dims(max_data, axis=axis)
                else:
                    max_reshaped = max_data
                    
                # The gradient flows only through the maximum elements
                mask = (self.data == max_reshaped)
                # Normalize the gradient if multiple elements achieve the maximum
                grad_scale = mask / np.sum(mask, axis=axis, keepdims=True)
                self.grad += grad_scale * (
                    np.expand_dims(out.grad, axis=axis) if axis is not None and not keepdims 
                    else out.grad
                )

            out.grad_fn = _max_backward
            out.set_requires_grad(True)

        return out
    
    def __len__(self) -> int:
        """
        Returns the length of the first dimension of the tensor.
        
        This is crucial for operations like CrossEntropyLoss where we need
        to iterate over batches or sequences. For a tensor of shape (batch_size, seq_len, features),
        len() would return batch_size.
        
        Returns:
            The size of the first dimension of the tensor
        """
        return self.shape[0]

    def __getitem__(self, idx) -> "Tensor":
        """
        Enables tensor indexing and slicing, similar to NumPy arrays.
        
        This method allows us to:
        1. Index into specific positions: tensor[0]
        2. Slice tensors: tensor[1:5]
        3. Use advanced indexing: tensor[indices]
        
        Args:
            idx: Index, slice, or array of indices
            
        Returns:
            A new tensor containing the indexed/sliced values
        """
        # Handle tuple indices for multiple dimensions
        if isinstance(idx, tuple):
            # Convert any Tensors in the index to their underlying data
            idx = tuple(i.data if isinstance(i, Tensor) else i for i in idx)
        elif isinstance(idx, Tensor):
            idx = idx.data

        out = Tensor(
            self.data[idx],
            dtype=self.dtype,
            _children=(self,),
            _op="getitem",
            requires_grad=self.requires_grad
        )

        if self.requires_grad and Tensor.grad_is_enabled:
            def _getitem_backward():
                grad = np.zeros_like(self.data)
                grad[idx] = out.grad
                self.grad += grad

            out.grad_fn = _getitem_backward
            out.set_requires_grad(True)

        return out