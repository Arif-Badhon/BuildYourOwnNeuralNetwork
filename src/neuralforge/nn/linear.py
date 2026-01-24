"""Linear layer - the fundamental neural network component."""

import numpy as np
from neuralforge.core.tensor import Tensor
from neuralforge.nn.module import Module


class Linear(Module):
    """Linear (fully connected) layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """Initialize Linear layer."""
        super().__init__()
    
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
    
        # He initialization - weight shape: (in_features, out_features)
        std = np.sqrt(2.0 / in_features)
        weight_data = np.random.normal(0, std, (in_features, out_features))
        weight = Tensor(weight_data.astype(np.float32), requires_grad=True)
        self.register_parameter("weight", weight)
    
        # Bias
        if bias:
            bias_data = np.zeros((out_features,), dtype=np.float32)
            bias_tensor = Tensor(bias_data, requires_grad=True)
            self.register_parameter("bias", bias_tensor)
        else:
            self._parameters["bias"] = None

    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: y = x @ W + b
    
        Instead of y = x @ W^T + b (which requires transposing),
        we just store weight as (in_features, out_features) and do x @ W directly.
        """
        weight = self._parameters["weight"]
        output = x @ weight
    
        if self.use_bias:
            output = output + self._parameters["bias"]
    
        return output






    
    def __repr__(self) -> str:
        """String representation."""
        bias_str = ", bias=True" if self.use_bias else ", bias=False"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"

