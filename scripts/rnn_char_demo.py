"""
Character-Level RNN Demo.

This script demonstrates how to train a Vanilla RNN to memorize a simple sequence.
Task: Given a character, predict the next character in "hello world".

This is a "Many-to-Many" task:
Input:  "h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"
Target: "e", "l", "l", "o", " ", "w", "o", "r", "l", "d", "!"
"""

import numpy as np
import sys
import os

# Add src to path so we can import neuralforge
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from neuralforge.core.tensor import Tensor
from neuralforge.nn.rnn import RNN
from neuralforge.nn.linear import Linear
from neuralforge.nn.loss import cross_entropy_loss
from neuralforge.optim.sgd import SGD
from typing import List, Dict

def one_hot_encode(char: str, vocab: Dict[str, int]) -> Tensor:
    """Create a One-Hot vector for a character."""
    vocab_size = len(vocab)
    vector = np.zeros((1, vocab_size), dtype=np.float32)
    vector[0, vocab[char]] = 1.0
    return Tensor(vector, requires_grad=False)

def main():
    print("=" * 60)
    print("NEURALFORGE - RNN CHARACTER DEMO")
    print("=" * 60)
    
    # 1. Prepare Data
    text = "hello world"
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    print(f"Text: '{text}'")
    print(f"Vocab: {chars} (Size: {vocab_size})")
    
    # Inputs: "hello world" (excluding last char for prediction usually, but here we do simple sequence)
    # Let's predict next char.
    # Input sequence: "hello worl"
    # Target sequence: "ello world"
    input_str = text[:-1]
    target_str = text[1:]
    
    # 2. Model Architecture
    hidden_size = 16
    rnn = RNN(input_size=vocab_size, hidden_size=hidden_size)
    output_layer = Linear(hidden_size, vocab_size)
    
    # Optimizer
    parameters = rnn.cell.parameters() + output_layer.parameters()
    optimizer = SGD(parameters, lr=0.05, momentum=0.9)
    
    print("\nTraining...")
    
    # 3. Training Loop
    epochs = 200
    
    for epoch in range(epochs):
        # A. Prepare Inputs (List of Tensors)
        inputs_list = [one_hot_encode(ch, char_to_idx) for ch in input_str]
        
        # B. Forward Pass (RNN)
        hidden_states, _ = rnn(inputs_list)
        
        # C. Compute Loss over the sequence
        total_loss = Tensor([0.0], requires_grad=True)
        predictions = ""
        
        for t, h_t in enumerate(hidden_states):
            logits = output_layer(h_t)
            target_char = target_str[t]
            target_tensor = one_hot_encode(target_char, char_to_idx)
            
            step_loss = cross_entropy_loss(logits, target_tensor)
            total_loss = total_loss + step_loss
            
            pred_idx = np.argmax(logits.data)
            predictions += idx_to_char[pred_idx]
            
        # D. Backward Pass
        optimizer.zero_grad()
        total_loss.backward()
        
        # E. Update
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Loss: {total_loss.item():.4f} | Pred: '{predictions}'")
            
    print("=" * 60)
    print(f"Final Prediction: '{predictions}'")
    print(f"Target Sequence : '{target_str}'")
    print("=" * 60)

if __name__ == "__main__":
    main()
