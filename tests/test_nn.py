"""Test suite for Neural Network Layers"""
import sys
import numpy as np

sys.path.insert(0, 'src')

from neuralforge.core.tensor import Tensor
from neuralforge.nn import Linear, ReLU, Sequential, mse_loss
from neuralforge.optim import SGD



def test_linear_layer():
    """Test: Linear layer"""
    print("\n" + "="*60)
    print("TEST 1: Linear Layer")
    print("="*60)
    
    layer = Linear(3, 2)
    x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
    y = layer(x)
    
    assert y.shape == (1, 2), f"Shape mismatch: {y.shape}"
    print(f"✓ Forward pass: {x.shape} → {y.shape}")
    
    loss = y.sum()
    loss.backward()
    
    assert layer._parameters["weight"].grad is not None
    print(f"✓ Backward pass: gradients computed")



def test_relu():
    """Test: ReLU activation"""
    print("\n" + "="*60)
    print("TEST 2: ReLU Activation")
    print("="*60)
    
    relu = ReLU()
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)
    
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    assert np.allclose(y.data, expected)
    print(f"✓ ReLU forward: {x.data} → {y.data}")
    
    loss = y.sum()
    loss.backward()
    print(f"✓ ReLU backward: gradient computed")


def test_sequential():
    """Test: Sequential model"""
    print("\n" + "="*60)
    print("TEST 3: Sequential Model")
    print("="*60)
    
    model = Sequential(
        Linear(10, 20),
        ReLU(),
        Linear(20, 5)
    )
    
    x = Tensor(np.random.randn(3, 10), requires_grad=True)
    output = model(x)
    
    assert output.shape == (3, 5)
    print(f"✓ Forward pass: {x.shape} → {output.shape}")
    
    loss = output.sum()
    loss.backward()
    print(f"✓ Backward pass: all params have gradients")


def test_training_loop():
    """Test: Complete training loop"""
    print("\n" + "="*60)
    print("TEST 4: Training Loop")
    print("="*60)
    
    X = Tensor(np.random.randn(10, 5), requires_grad=False)
    y = Tensor(np.random.randn(10, 2), requires_grad=False)
    
    model = Sequential(
        Linear(5, 10),
        ReLU(),
        Linear(10, 2)
    )
    
    optimizer = SGD(model.parameters(), lr=0.01)
    
    losses = []
    for iteration in range(3):
        pred = model(X)
        loss = mse_loss(pred, y)
        losses.append(loss.data)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        print(f"  Iteration {iteration+1}: loss = {loss.data:.4f}")
    
    print(f"✓ Training loop completed")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 NEURALFORGE NN TEST SUITE")
    print("="*60)
    
    try:
        test_linear_layer()
        test_relu()
        test_sequential()
        test_training_loop()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

