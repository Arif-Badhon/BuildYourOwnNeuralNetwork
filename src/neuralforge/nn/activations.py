"""Activation functions for neural networks."""

import numpy as np
from neuralforge.core.tensor import Tensor
from neuralforge.nn.module import Module


class ReLU(Module):
    """Rectified Linear Unit activation."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply ReLU: max(0, x)."""
        return x.relu()


class Sigmoid(Module):
    """Sigmoid activation function."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Sigmoid."""
        return x.sigmoid()


class Tanh(Module):
    """Hyperbolic tangent activation."""
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Tanh."""
        return x.tanh()


class Softmax(Module):
    """Softmax activation function."""
    
    def __init__(self, dim: int = -1):
        """Initialize Softmax."""
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply Softmax."""
        return x.softmax(dim=self.dim)


class LeakyReLU(Module):
    """Leaky ReLU activation."""
    
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply LeakyReLU."""
        positive = x * (x.data > 0).astype(np.float32)
        negative = x * (x.data <= 0).astype(np.float32) * self.negative_slope
        return positive + negative

