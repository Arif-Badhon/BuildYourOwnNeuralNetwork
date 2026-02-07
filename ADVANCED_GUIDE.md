# NeuralForge: Advanced User Guide 🧠🛠️

Welcome to the deep dive into **NeuralForge**. This guide is designed for machine learning professionals and enthusiasts who want to understand the *internals* of a deep learning library. We will explore how the autograd engine works, how to implement custom operations, and how to extend the library.

---

## 1. The Core: Autograd Engine

At the heart of NeuralForge is the `Tensor` class (`src/neuralforge/core/tensor.py`). Unlike standard NumPy arrays, a `Tensor` is a node in a dynamic computation graph.

### The `Tensor` Object

key attributes of a `Tensor`:
- **`data`**: The underlying NumPy array holding the values.
- **`grad`**: A NumPy array holding the gradients (same shape as `data`). Populated during `backward()`.
- **`_children`**: A set of `Tensor` objects that were used to create this tensor. This defines the edges of the DAG (Directed Acyclic Graph).
- **`_op`**: A string representing the operation that created this tensor (e.g., `'add'`, `'mul'`, `'sigmoid'`).
- **`_backward`**: A closure (function) that knows how to compute the gradient of the *parents* (inputs to the operation) given the gradient of this tensor.

### Reverse-Mode Autograd (Backpropagation)

When you call `.backward()` on a scalar loss tensor:

1.  **Topological Sort**: We perform a DFS to physically order the nodes in the graph so that we process children before parents.
2.  **Gradient Accumulation**: We iterate efficiently through the sorted nodes in reverse order. For each node, we call its `_backward()` function.
3.  **Chain Rule**: The `_backward()` function applies the chain rule, multiplying the current tensor's `grad` (dL/dOutput) by the local derivative (dOutput/dInput) and accumulating the result into the input's `grad`.

```python
# Pseudo-implementation of _backward for addition (z = x + y)
def _backward():
    # dL/dx = dL/dz * dz/dx = grad * 1
    self.grad += out.grad 
    # dL/dy = dL/dz * dz/dy = grad * 1
    other.grad += out.grad
```

---

## 2. Implementing Custom Operations

You can easily extend NeuralForge with new operations. You just need to define the forward pass and the backward pass closure.

### Example: LeakyReLU

Let's implement LeakyReLU: $f(x) = x$ if $x > 0$, else $\alpha x$.

```python
from neuralforge.core.tensor import Tensor
import numpy as np

def leaky_relu(self, alpha=0.01):
    # 1. Forward Pass
    out_data = np.where(self.data > 0, self.data, self.data * alpha)
    
    # 2. Create Output Tensor (connects to self in the graph)
    out = Tensor(
        out_data, 
        _children=(self,), 
        _op=f'leaky_relu'
    )
    
    # 3. Define Backward Pass
    def _backward():
        if self.requires_grad:
            # Gradient is 1 where x > 0, and alpha where x <= 0
            grad_fn = np.where(self.data > 0, 1, alpha)
            self.grad += out.grad * grad_fn
            
    out._backward = _backward
    return out

# Monkey-patch it onto Tensor (or subclass Tensor)
Tensor.leaky_relu = leaky_relu
```

---

## 3. The `Module` Abstraction

The `nn.Module` class (`src/neuralforge/nn/module.py`) provides a higher-level abstraction for organizing parameters.

- **`register_parameter(name, tensor)`**: Adds a tensor to `self._parameters`.
- **`parameters()`**: Recursively collects all parameters from the module and its submodules.
- **`zero_grad()`**: Clears gradients for all parameters.

This design mirrors PyTorch, making it intuitive to build complex architectures like RNNs or Transformers by nesting modules.

---

## 4. Advanced: Recurrent Neural Networks (RNNs)

NeuralForge supports dynamic graphs, which makes RNNs straightforward to implement via Python loops (no static graph compilation needed).

Check `src/neuralforge/nn/rnn.py` for a clean implementation of a "Vanilla" RNN. It explicitly demonstrates "Unrolling" the computation graph over time:

```python
# Conceptual loop
for x_t in inputs:
    # h_t depends on x_t and h_{t-1}
    # This builds a deep computation graph graph connected through time!
    h_t = self.cell(x_t, h_t) 
```

---

## 5. Contributing & Extensions

Ready to contribute? Here are some advanced challenges:

1.  **Optimizers**: Implement `Adam` or `RMSprop` in `src/neuralforge/optim`.
2.  **Layers**: Implement `Conv2d` (requires `im2col` for efficiency) or `Dropout`.
3.  **Losses**: Implement `CrossEntropyLoss` with `LogSoftmax` for numerical stability.
4.  **Broadcasting**: Enhance core operations to handle broadcasting automatically during the backward pass (summing out broadcasted dimensions).
