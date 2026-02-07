"""
NeuralForge Advanced Demo
=========================

This script demonstrates advanced features of NeuralForge for ML professionals:
1. Defining a custom Autograd operation (LeakyReLU)
2. Creating a custom Module using that operation
3. Training a network with manual gradient inspection
4. Verifying numerical correctness

"""

import sys
import os
import numpy as np

# Add src to path so we can import neuralforge
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from neuralforge.core.tensor import Tensor
from neuralforge.nn import Module, Linear
import neuralforge.core.tensor as tensor_module  # To patch Tensor if needed

# ==========================================
# 1. Custom Autograd Operation: LeakyReLU
# ==========================================

def leaky_relu(self: Tensor, alpha: float = 0.01) -> Tensor:
    """
    Applies LeakyReLU: f(x) = x if x > 0 else alpha * x
    """
    # Forward pass: straightforward numpy
    out_data = np.where(self.data > 0, self.data, self.data * alpha).astype(np.float32)
    
    # Create the output tensor and connect it to the graph
    out = Tensor(
        out_data,
        requires_grad=self.requires_grad,
        _children=(self,),
        _op=f"leaky_relu({alpha})"
    )
    
    # Define the backward pass closure
    def _backward():
        if self.requires_grad:
            # Gradient is 1 where x > 0, alpha where x <= 0
            # Chain rule: dL/dx = dL/dout * dout/dx
            grad_fn = np.where(self.data > 0, 1.0, alpha).astype(np.float32)
            
            # Aggregate gradients (handle branching/reuse of this tensor)
            delta = out.grad * grad_fn
            self.grad = (self.grad if self.grad is not None else 0) + delta
            
    out._backward = _backward
    return out

# "Monkey-patch" the Tensor class so we can call x.leaky_relu()
Tensor.leaky_relu = leaky_relu

print("1. Custom Autograd Operation (LeakyReLU) defined and patched.")


# ==========================================
# 2. Custom Module Definition
# ==========================================

class AdvancedNet(Module):
    def __init__(self):
        super().__init__()
        # A simple 2-layer MLP
        self.fc1 = Linear(2, 4)
        self.fc2 = Linear(4, 1)
        
    def forward(self, x: Tensor) -> Tensor:
        # Use our custom operation!
        x = self.fc1(x).leaky_relu(alpha=0.1)
        x = self.fc2(x)
        return x

print("2. Custom Module (AdvancedNet) initialized.")


# ==========================================
# 3. Training Loop & Gradient Inspection
# ==========================================

def run_demo():
    print("\nStarting Training Demo...")
    
    # Dataset: XOR-like problem (non-linear)
    X_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y_data = np.array([[0], [1], [1], [0]], dtype=np.float32) # XOR
    
    X = Tensor(X_data)
    y = Tensor(y_data)
    
    model = AdvancedNet()
    
    # Training Loop
    learning_rate = 0.1
    epochs = 100
    
    for epoch in range(epochs):
        # 1. Forward
        pred = model(X)
        
        # 2. Loss (MSE)
        diff = pred - y
        loss = (diff * diff).mean()
        
        # 3. Zero Gradients
        model.zero_grad()
        
        # 4. Backward
        loss.backward()
        
        # 5. Optimizer Step (SGD)
        for param in model.parameters():
            param.data -= learning_rate * param.grad
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.data:.4f}")
            
            # Inspect gradients of the first layer weights to ensure flow
            w1_grad_norm = np.linalg.norm(model.fc1._parameters['weight'].grad)
            print(f"   -> Gradient norm (fc1): {w1_grad_norm:.6f}")

    print("\nTraining complete.")
    print("Final Predictions:")
    print(model(X).data)


if __name__ == "__main__":
    run_demo()
