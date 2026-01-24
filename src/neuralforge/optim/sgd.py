"""SGD (Stochastic Gradient Descent) optimizer."""

from typing import List
from neuralforge.core.tensor import Tensor



class SGD:
    """Stochastic Gradient Descent optimizer."""
    
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0
    ):
        """Initialize SGD optimizer."""
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}
        
        for i, param in enumerate(self.params):
            self.velocity[i] = None
    
    def step(self) -> None:
        """Update parameters based on their gradients."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # L2 regularization
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # Apply momentum
            if self.momentum != 0:
                if self.velocity[i] is None:
                    self.velocity[i] = -self.lr * grad
                else:
                    self.velocity[i] = (
                        self.momentum * self.velocity[i] - self.lr * grad
                    )
                param.data = param.data + self.velocity[i]
            else:
                param.data = param.data - self.lr * grad
    
    def zero_grad(self) -> None:
        """Reset all parameter gradients."""
        for param in self.params:
            param.grad = None
    
    def set_lr(self, new_lr: float) -> None:
        """Change learning rate."""
        if new_lr < 0.0:
            raise ValueError(f"Invalid learning rate: {new_lr}")
        self.lr = new_lr

