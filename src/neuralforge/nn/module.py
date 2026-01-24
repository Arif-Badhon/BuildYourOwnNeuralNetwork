"""
Base Module class for all neural network components.
"""

from typing import Dict, List
from neuralforge.core.tensor import Tensor


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        """Initialize module."""
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, 'Module'] = {}
    
    def forward(self, *args, **kwargs) -> Tensor:
        """Forward pass - override in subclasses."""
        raise NotImplementedError(
            f"Module {self.__class__.__name__} does not implement forward()"
        )
    
    def __call__(self, *args, **kwargs) -> Tensor:
        """Makes module callable."""
        return self.forward(*args, **kwargs)
    
    def register_parameter(self, name: str, param: Tensor) -> None:
        """Register a parameter (trainable weight)."""
        if not isinstance(param, Tensor):
            raise TypeError(f"Parameter must be Tensor, got {type(param)}")
        self._parameters[name] = param
    
    def register_module(self, name: str, module: 'Module') -> None:
        """Register a submodule."""
        if not isinstance(module, Module):
            raise TypeError(f"Module must be Module, got {type(module)}")
        self._modules[name] = module
    
    def parameters(self) -> List[Tensor]:
        """Get all trainable parameters."""
        params = []
        params.extend(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self) -> None:
        """Reset all gradients to None."""
        for param in self._parameters.values():
            param.grad = None
        for module in self._modules.values():
            module.zero_grad()
    
    def train(self) -> None:
        """Set module to training mode."""
        self.training = True
        for module in self._modules.values():
            module.train()
    
    def eval(self) -> None:
        """Set module to evaluation mode."""
        self.training = False
        for module in self._modules.values():
            module.eval()
    
    def __repr__(self) -> str:
        """String representation."""
        main_str = self.__class__.__name__ + "("
        params_str = ", ".join(
            f"{name}={param.shape}" for name, param in self._parameters.items()
        )
        if params_str:
            main_str += params_str
        main_str += ")"
        return main_str

