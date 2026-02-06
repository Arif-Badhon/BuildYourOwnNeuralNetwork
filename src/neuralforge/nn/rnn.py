"""
Recurrent Neural Network (RNN) implementation.

This module demonstrates how sequences are processed in Deep Learning.
Unlike Feed-Forward networks (Linear layers), RNNs maintain 'state' (hidden state)
that evolves as we process a sequence step-by-step.
"""

import numpy as np
from typing import List, Optional, Tuple
from neuralforge.core.tensor import Tensor
from neuralforge.nn.module import Module

class RNNCell(Module):
    """
    A single time-step of a Vanilla RNN.
    
    Formula: h_t = tanh(x_t @ W_ih + b_ih + h_{t-1} @ W_hh + b_hh)
    
    This is the "atomic unit" of an RNN.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # 1. Weights for Input -> Hidden
        # "He" initialization helps with training stability
        std = np.sqrt(2.0 / (input_size + hidden_size))
        
        W_ih = np.random.normal(0, std, (input_size, hidden_size)).astype(np.float32)
        self.register_parameter("weight_ih", Tensor(W_ih, requires_grad=True))
        
        # 2. Weights for Hidden -> Hidden
        W_hh = np.random.normal(0, std, (hidden_size, hidden_size)).astype(np.float32)
        self.register_parameter("weight_hh", Tensor(W_hh, requires_grad=True))
        
        # 3. Biases
        if bias:
            b_ih = np.zeros((hidden_size,), dtype=np.float32)
            b_hh = np.zeros((hidden_size,), dtype=np.float32)
            self.register_parameter("bias_ih", Tensor(b_ih, requires_grad=True))
            self.register_parameter("bias_hh", Tensor(b_hh, requires_grad=True))
        else:
            self._parameters["bias_ih"] = None
            self._parameters["bias_hh"] = None
            
    def forward(self, input: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """
        Compute one step.
        
        Args:
            input: Tensor of shape (batch, input_size)
            hidden: Tensor of shape (batch, hidden_size). If None, init to zeros.
            
        Returns:
            next_hidden: Tensor of shape (batch, hidden_size)
        """
        # 1. Handle initial hidden state
        if hidden is None:
            batch_size = input.shape[0]
            hidden = Tensor(
                np.zeros((batch_size, self.hidden_size), dtype=np.float32),
                requires_grad=True
            )
            
        # 2. Compute Input contribution: x_t @ W_ih + b_ih
        ih = input @ self._parameters["weight_ih"]
        if self.bias:
            ih = ih + self._parameters["bias_ih"]
            
        # 3. Compute Hidden contribution: h_{t-1} @ W_hh + b_hh
        hh = hidden @ self._parameters["weight_hh"]
        if self.bias:
            hh = hh + self._parameters["bias_hh"]
            
        # 4. Combine and Activate
        # We assume tanh activation for vanilla RNN
        next_hidden = (ih + hh).tanh()
        
        return next_hidden
    
    def __repr__(self) -> str:
        return f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class RNN(Module):
    """
    A multi-step RNN that unrolls the loop over the sequence.
    
    This Module takes a sequence of inputs and processes them in order,
    passing the hidden state from one step to the next.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size, bias)
        self.register_module("cell", self.cell)
        
    def forward(self, inputs: List[Tensor], hidden: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Process a full sequence.
        
        Args:
            inputs: List of Tensors, each of shape (batch, input_size).
                    We use a List instead of a 3D Tensor to make the "unrolling" explicit
                    and avoid complex slicing logic in specific Tensor implementations.
            hidden: Initial hidden state (batch, hidden_size)
            
        Returns:
            all_hidden_states: Tensor of shape (batch, seq_len, hidden_size) 
                               (Stacked outputs from all steps)
            final_hidden_state: Tensor of shape (batch, hidden_size)
                                (The state after the last step)
        """
        hidden_states = []
        h_t = hidden
        
        # TIME LOOP: The "Recurrent" part
        for x_t in inputs:
            h_t = self.cell(x_t, h_t)
            hidden_states.append(h_t)
            
        # Stack hidden states into one tensor (optional, if we had stack support)
        # For now, we returns the list of states or the final state mostly.
        # But to be compatible with standard APIs, we often return all.
        # Since we don't have 'stack' in Tensor yet, let's return the list for now
        # or just the final state if that's what we need for classification.
        
        # For the demo, we usually need the output at every step (many-to-many)
        # or just the last (many-to-one).
        
        return hidden_states, h_t # type: ignore
    
    def __repr__(self) -> str:
        return f"RNN(input_size={self.cell.input_size}, hidden_size={self.hidden_size})"
