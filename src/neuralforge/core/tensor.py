"""
Core Tensor implementation for NeuralForge - Step 1 of Building Neural Networks from Scratch.

This module provides the fundamental Tensor class that:
1. Stores numerical data (forward values)
2. Tracks gradients (backward values)  
3. Maintains computation graph for backpropagation
4. Implements basic operations with automatic differentiation

CONCEPT: Every neural network is ultimately a composition of tensor operations.
By implementing this ourselves, we understand exactly how PyTorch/TensorFlow work internally.
"""

import numpy as np
from typing import Union, Callable, Tuple, Optional


class Tensor:
    """
    A multi-dimensional array that tracks computation graph for automatic differentiation.
    
    This is the CORE of any deep learning framework. Think of it as:
    - A smart numpy array that remembers how it was created
    - A node in a computation graph
    - A container for both values AND gradients
    
    Attributes:
        data (np.ndarray): The actual numerical values
        grad (np.ndarray): Gradients computed during backward pass
        requires_grad (bool): Track gradients for this tensor?
        _children (set): Which tensors created this tensor
        _op (str): What operation created this tensor (for debugging)
        _backward (Callable): Function to compute gradients during backprop
    """
    
    def __init__(
        self,
        data: Union[float, list, np.ndarray],
        requires_grad: bool = False,
        _children: Tuple = (),
        _op: str = "",
    ):
        """
        Initialize a Tensor.
        
        Args:
            data: Input data (automatically converted to numpy array)
            requires_grad: Whether to compute/track gradients for this tensor
            _children: Child tensors in computation graph (used internally)
            _op: Operation name string (used internally for debugging)
        """
        # Store data as float32 array (standard for neural networks)
        self.data = np.array(data, dtype=np.float32)
        
        # Gradients initialized as None (computed during backward pass)
        self.grad: Optional[np.ndarray] = None
        self.requires_grad = requires_grad
        
        # Computation graph: track which tensors created this one
        self._children = set(_children)
        self._op = _op
        
        # Function to execute during backward pass (filled by operations)
        self._backward: Callable = lambda: None
    
    # ================== PROPERTIES & INTROSPECTION ==================
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Return shape of underlying data (e.g., (batch_size, features))."""
        return self.data.shape
    
    @property
    def dtype(self):
        """Return numpy dtype of data."""
        return self.data.dtype
    
    @property
    def size(self) -> int:
        """Return total number of elements in tensor."""
        return self.data.size
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, requires_grad={self.requires_grad})"
    
    # ================== ARITHMETIC OPERATIONS ==================
    
    def __add__(self, other: "Tensor") -> "Tensor":
        """
        Element-wise addition: c = a + b
        
        GRADIENTS (via chain rule):
        - d(c)/d(a) = 1  →  a.grad += c.grad
        - d(c)/d(b) = 1  →  b.grad += c.grad
        """
        other = _ensure_tensor(other)
        out = Tensor(
            self.data + other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="add"
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) + out.grad
        
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        """Handle: scalar + Tensor"""
        return self + other
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        """
        Element-wise subtraction: c = a - b
        
        GRADIENTS:
        - d(c)/d(a) = 1   →  a.grad += c.grad
        - d(c)/d(b) = -1  →  b.grad -= c.grad  (note the negative!)
        """
        other = _ensure_tensor(other)
        out = Tensor(
            self.data - other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="sub"
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad
            if other.requires_grad:
                other.grad = (other.grad if other.grad is not None else 0) - out.grad
        
        out._backward = _backward
        return out
    
    def __rsub__(self, other):
        """Handle: scalar - Tensor"""
        return _ensure_tensor(other) - self
    
    def __mul__(self, other: "Tensor") -> "Tensor":
        """
        Element-wise multiplication: c = a * b
        
        GRADIENTS (product rule from calculus):
        - d(c)/d(a) = b   →  a.grad += b * c.grad
        - d(c)/d(b) = a   →  b.grad += a * c.grad
        
        This is the KEY insight: gradient depends on the OTHER operand!
        """
        other = _ensure_tensor(other)
        out = Tensor(
            self.data * other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="mul"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * other.data
                self.grad = (self.grad if self.grad is not None else 0) + grad
            if other.requires_grad:
                grad = out.grad * self.data
                other.grad = (other.grad if other.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        """Handle: scalar * Tensor"""
        return self * other
    
    def __truediv__(self, other: "Tensor") -> "Tensor":
        """
        Element-wise division: c = a / b
        
        GRADIENTS (quotient rule):
        - d(c)/d(a) = 1/b        →  a.grad += c.grad / b
        - d(c)/d(b) = -a/(b²)    →  b.grad -= a * c.grad / (b²)
        """
        other = _ensure_tensor(other)
        out = Tensor(
            self.data / other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="div"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad / other.data
                self.grad = (self.grad if self.grad is not None else 0) + grad
            if other.requires_grad:
                grad = out.grad * (-self.data / (other.data ** 2))
                other.grad = (other.grad if other.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def __rtruediv__(self, other):
        """Handle: scalar / Tensor"""
        return _ensure_tensor(other) / self
    
    def __pow__(self, power: Union[int, float]) -> "Tensor":
        """
        Element-wise exponentiation: c = a ** power
        
        GRADIENTS (power rule):
        - d(c)/d(a) = power * a^(power-1)
        
        Example: if a²: d/da = 2*a
        """
        out = Tensor(
            self.data ** power,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op=f"pow({power})"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (power * self.data ** (power - 1))
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def __neg__(self) -> "Tensor":
        """Negation: -a = a * (-1)"""
        return self * -1
    
    # ================== MATRIX OPERATIONS ==================
    
    def __matmul__(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication: C = A @ B
        
        Shape: (m, n) @ (n, p) = (m, p)
        
        GRADIENTS (chain rule with matrix derivatives):
        - dL/dA = dL/dC @ B^T
        - dL/dB = A^T @ dL/dC
        
        This is THE most important operation in neural networks!
        """
        other = _ensure_tensor(other)
        out = Tensor(
            self.data @ other.data,
            requires_grad=self.requires_grad or other.requires_grad,
            _children=(self, other),
            _op="matmul"
        )
        
        def _backward():
            if self.requires_grad:
                other_T = np.swapaxes(other.data, -2, -1)
                grad = out.grad @ other_T
                self.grad = (self.grad if self.grad is not None else 0) + grad
            if other.requires_grad:
                self_T = np.swapaxes(self.data, -2, -1)
                grad = self_T @ out.grad
                other.grad = (other.grad if other.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def __rmatmul__(self, other):
        """Handle: scalar @ Tensor"""
        return _ensure_tensor(other) @ self
    
    def transpose(self, axes: Tuple[int, int] = (-2, -1)) -> "Tensor":
        """
        Transpose tensor along specified axes.
        
        Default: transpose last two dimensions (typical for matrices)
        """
        # Normalize negative indices
        if axes[0] < 0:
            axes = (len(self.shape) + axes[0], axes[1])
        if axes[1] < 0:
            axes = (axes[0], len(self.shape) + axes[1])
        
        out = Tensor(
            np.transpose(self.data, axes),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="transpose"
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + np.transpose(out.grad, axes)
        
        out._backward = _backward
        return out
    
    @property
    def T(self) -> "Tensor":
        """Shorthand: tensor.T = tensor.transpose()"""
        return self.transpose()
    
    # ================== ACTIVATION FUNCTIONS ==================
    # These are non-linear transformations crucial for neural networks!
    
    def relu(self) -> "Tensor":
        """
        ReLU (Rectified Linear Unit): max(0, x)
        
        WHY? Introduces non-linearity. Without it, stacking layers = single layer.
        
        GRADIENT:
        - If x > 0: d/dx = 1
        - If x ≤ 0: d/dx = 0  (gradient "dies" for negative inputs)
        """
        out = Tensor(
            np.maximum(0, self.data),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="relu"
        )
        
        def _backward():
            if self.requires_grad:
                mask = (self.data > 0).astype(np.float32)
                self.grad = (self.grad if self.grad is not None else 0) + out.grad * mask
        
        out._backward = _backward
        return out
    
    def sigmoid(self) -> "Tensor":
        """
        Sigmoid: σ(x) = 1 / (1 + e^(-x))
        
        Maps output to (0, 1) - useful for binary classification
        
        GRADIENT: d/dx = σ(x) * (1 - σ(x))
        
        Note: Numerically stable implementation to avoid overflow
        """
        # For numerical stability, use different formula for pos/neg
        sigmoid_data = np.where(
            self.data >= 0,
            1 / (1 + np.exp(-self.data)),
            np.exp(self.data) / (1 + np.exp(self.data))
        ).astype(np.float32)
        
        out = Tensor(
            sigmoid_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sigmoid"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * out.data * (1 - out.data)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def tanh(self) -> "Tensor":
        """
        Tanh: tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        
        Maps output to (-1, 1) - smoother gradient than sigmoid
        
        GRADIENT: d/dx = 1 - tanh²(x)
        """
        tanh_data = np.tanh(self.data).astype(np.float32)
        
        out = Tensor(
            tanh_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="tanh"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad * (1 - out.data ** 2)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def softmax(self, axis: int = -1) -> "Tensor":
        """
        Softmax: softmax(x_i) = e^(x_i) / Σ(e^(x_j))
        
        Maps outputs to probability distribution: sum=1, each in (0,1)
        Used with cross-entropy for multi-class classification
        
        GRADIENT: Complex Jacobian, simplified here for common case
        """
        # Numerical stability: subtract max before exponential
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax_data = (exp_data / np.sum(exp_data, axis=axis, keepdims=True)).astype(np.float32)
        
        out = Tensor(
            softmax_data,
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="softmax"
        )
        
        def _backward():
            if self.requires_grad:
                # Jacobian: diag(p) - p*p^T, but simplified for typical usage
                s = out.data
                grad = out.grad * (s * (1 - s))
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    # ================== REDUCTION OPERATIONS ==================
    
    def sum(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """
        Sum reduction: σ(x_i)
        
        Args:
            axis: Which axis to sum over (None = sum all)
            keepdims: Keep the dimension (size=1) or remove it?
        
        GRADIENT: Gradient is broadcast back to original shape
        """
        out = Tensor(
            np.sum(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="sum"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.data.shape)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    def mean(self, axis: Optional[int] = None, keepdims: bool = False) -> "Tensor":
        """
        Mean reduction: (1/n) * Σ(x_i)
        
        GRADIENT: Same as sum, but divided by n
        """
        if axis is None:
            n = self.data.size
        else:
            n = self.data.shape[axis]
        
        out = Tensor(
            np.mean(self.data, axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="mean"
        )
        
        def _backward():
            if self.requires_grad:
                grad = out.grad / n
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, self.data.shape)
                self.grad = (self.grad if self.grad is not None else 0) + grad
        
        out._backward = _backward
        return out
    
    # ================== BACKPROPAGATION ==================
    
    def backward(self, gradient: Optional[np.ndarray] = None):
        """
        Compute gradients via REVERSE-MODE AUTOMATIC DIFFERENTIATION.
        
        This is the heart of deep learning! Algorithm:
        
        1. Start with gradient of output (usually 1 or dL/dOutput)
        2. Traverse computation graph BACKWARDS
        3. At each node: apply chain rule to compute gradient
        4. Accumulate gradients for tensors with requires_grad=True
        
        EXAMPLE:
            a = Tensor([2.0], requires_grad=True)
            b = Tensor([3.0], requires_grad=True)
            c = a * b           # c = 6.0
            d = c + a           # d = 8.0
            d.backward()        # Compute gradients!
            
            # Gradients computed via chain rule:
            # dL/da = dL/dd * dd/da = 1 * (b + 1) = 4.0
            # dL/db = dL/dd * dd/dc * dc/db = 1 * 1 * a = 2.0
        
        Args:
            gradient: Initial gradient (default: ones - for scalar outputs)
        """
        if gradient is None:
            gradient = np.ones_like(self.data, dtype=np.float32)
        
        self.grad = gradient
        
        # Build topological order: which tensors to process in what order?
        # Use DFS to ensure dependencies are resolved before a node
        visited = set()
        topo_order = []
        
        def build_topo(tensor):
            if id(tensor) in visited:
                return
            visited.add(id(tensor))
            for child in tensor._children:
                build_topo(child)
            topo_order.append(tensor)
        
        build_topo(self)
        
        # Reverse pass: apply chain rule starting from output
        for tensor in reversed(topo_order):
            tensor._backward()  # Each node computes gradient of its inputs
    
    def zero_grad(self):
        """Reset gradients to None (essential for training loops!)"""
        self.grad = None
    
    # ================== UTILITY METHODS ==================
    
    def item(self) -> float:
        """
        Extract scalar value from single-element tensor.
        
        Raises ValueError if tensor has more than one element.
        """
        if self.data.size != 1:
            raise ValueError(f"Cannot extract item from tensor with {self.data.size} elements")
        return float(self.data.flat[0])
    
    def detach(self) -> "Tensor":
        """
        Break computation graph: return tensor without gradient tracking.
        
        Useful when you want to use tensor values but don't want gradients.
        """
        detached = Tensor(self.data.copy(), requires_grad=False)
        detached.grad = self.grad.copy() if self.grad is not None else None
        return detached
    
    def clone(self) -> "Tensor":
        """Create independent copy of this tensor."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)
    
    def reshape(self, shape: Tuple[int, ...]) -> "Tensor":
        """Change shape without changing data (e.g., (6,) to (2, 3))."""
        out = Tensor(
            self.data.reshape(shape),
            requires_grad=self.requires_grad,
            _children=(self,),
            _op="reshape"
        )
        
        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + out.grad.reshape(self.data.shape)
        
        out._backward = _backward
        return out
    
    def flatten(self) -> "Tensor":
        """Reshape to 1D array."""
        return self.reshape((-1,))


# ================== HELPER FUNCTIONS ==================

def _ensure_tensor(obj: Union[Tensor, float, int, list, np.ndarray]) -> Tensor:
    """
    Convert any object to Tensor for uniform handling.
    
    This allows operations like:
        Tensor([1.0, 2.0]) + 5  # 5 becomes Tensor([5.0, 5.0])
    """
    if isinstance(obj, Tensor):
        return obj
    return Tensor(obj, requires_grad=False)


# ================== LOSS FUNCTIONS ==================

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Squared Error: (1/n) * Σ(y_pred - y_true)²
    
    Common for regression tasks. Penalizes large errors heavily.
    """
    diff = predictions - targets
    squared = diff * diff
    return squared.mean()


def mae_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """
    Mean Absolute Error: (1/n) * Σ|y_pred - y_true|
    
    More robust to outliers than MSE.
    """
    diff = predictions - targets
    abs_diff = (diff * diff + 1e-7) ** 0.5  # sqrt(x²) ≈ |x|
    return abs_diff.mean()


if __name__ == "__main__":
    print("=" * 60)
    print("NEURALFORGE - STEP 1: TENSOR OPERATIONS")
    print("=" * 60)
    
    # Test 1: Basic arithmetic
    print("\n✓ Test 1: Basic Arithmetic Operations")
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    print(f"  a + b = {c.data}")
    
    # Test 2: Gradient computation
    print("\n✓ Test 2: Gradient Computation (Backpropagation)")
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    d = c.sum()
    d.backward()
    print(f"  a.grad = {a.grad}  (should be b's values)")
    print(f"  b.grad = {b.grad}  (should be a's values)")
    
    # Test 3: Matrix multiplication
    print("\n✓ Test 3: Matrix Multiplication")
    A = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    B = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    C = A @ B
    print(f"  Shape: {A.shape} @ {B.shape} = {C.shape}")
    
    # Test 4: Activation functions
    print("\n✓ Test 4: Activation Functions")
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    relu_out = x.relu()
    print(f"  ReLU({x.data}) = {relu_out.data}")
    
    sigmoid_out = x.sigmoid()
    print(f"  Sigmoid({x.data}) = {sigmoid_out.data}")
    
    print("\n" + "=" * 60)
    print("All tests passed! Ready to build neural networks.")
    print("=" * 60)