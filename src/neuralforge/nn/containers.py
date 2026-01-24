"""Container modules for composing neural network layers."""

from neuralforge.core.tensor import Tensor
from neuralforge.nn.module import Module



class Sequential(Module):
    """Sequential container - runs modules one after another."""
    
    def __init__(self, *modules: Module):
        """Initialize Sequential container."""
        super().__init__()
        for i, module in enumerate(modules):
            self.register_module(str(i), module)
    
    def forward(self, x: Tensor) -> Tensor:
        """Run input through each module in sequence."""
        for module in self._modules.values():
            x = module(x)
        return x
    
    def __repr__(self) -> str:
        """String representation."""
        lines = ["Sequential("]
        for name, module in self._modules.items():
            lines.append(f"  ({name}): {repr(module)}")
        lines.append(")")
        return "\n".join(lines)


class ModuleList(Module):
    """Unordered list of modules."""
    
    def __init__(self, modules=None):
        """Initialize ModuleList."""
        super().__init__()
        if modules is None:
            modules = []
        for i, module in enumerate(modules):
            self.register_module(str(i), module)
    
    def append(self, module: Module) -> 'ModuleList':
        """Add a module to the list."""
        self.register_module(str(len(self._modules)), module)
        return self
    
    def __getitem__(self, index: int) -> Module:
        """Get module by index."""
        return self._modules[str(index)]
    
    def __len__(self) -> int:
        """Get number of modules."""
        return len(self._modules)
    
    def __iter__(self):
        """Iterate over modules."""
        for i in range(len(self._modules)):
            yield self._modules[str(i)]
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass not defined."""
        raise NotImplementedError("ModuleList does not define forward()")


class ModuleDict(Module):
    """Dictionary of named modules."""
    
    def __init__(self, modules=None):
        """Initialize ModuleDict."""
        super().__init__()
        if modules is None:
            modules = {}
        for name, module in modules.items():
            self.register_module(name, module)
    
    def __getitem__(self, key: str) -> Module:
        """Get module by name."""
        return self._modules[key]
    
    def __len__(self) -> int:
        """Get number of modules."""
        return len(self._modules)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass not defined."""
        raise NotImplementedError("ModuleDict does not define forward()")

