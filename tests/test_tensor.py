"""
STEP 1: Test suite for Tensor implementation
Run this to verify your tensor works correctly!
"""

import sys
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

from neuralforge.core.tensor import Tensor, mse_loss


def test_basic_arithmetic():
    """Test: Basic addition, subtraction, multiplication, division"""
    print("\n" + "="*60)
    print("TEST 1: Basic Arithmetic Operations")
    print("="*60)
    
    # Addition
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    c = a + b
    assert np.allclose(c.data, [5.0, 7.0, 9.0]), "Addition failed"
    print("✓ Addition: [1,2,3] + [4,5,6] = [5,7,9]")
    
    # Subtraction
    d = a - b
    assert np.allclose(d.data, [-3.0, -3.0, -3.0]), "Subtraction failed"
    print("✓ Subtraction: [1,2,3] - [4,5,6] = [-3,-3,-3]")
    
    # Multiplication
    e = a * b
    assert np.allclose(e.data, [4.0, 10.0, 18.0]), "Multiplication failed"
    print("✓ Multiplication: [1,2,3] * [4,5,6] = [4,10,18]")
    
    # Division
    f = b / a
    assert np.allclose(f.data, [4.0, 2.5, 2.0]), "Division failed"
    print("✓ Division: [4,5,6] / [1,2,3] = [4,2.5,2]")


def test_gradients_addition():
    """Test: Gradients for addition"""
    print("\n" + "="*60)
    print("TEST 2: Gradients - Addition")
    print("="*60)
    
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()
    
    assert np.allclose(a.grad, [1.0, 1.0]), f"a.grad = {a.grad}, expected [1,1]"
    assert np.allclose(b.grad, [1.0, 1.0]), f"b.grad = {b.grad}, expected [1,1]"
    print("✓ Addition gradients computed correctly")
    print(f"  Loss = sum(a + b)")
    print(f"  a.grad = {a.grad}")
    print(f"  b.grad = {b.grad}")


def test_gradients_multiplication():
    """Test: Gradients for multiplication (product rule)"""
    print("\n" + "="*60)
    print("TEST 3: Gradients - Multiplication (Product Rule)")
    print("="*60)
    
    a = Tensor([2.0], requires_grad=True)
    b = Tensor([3.0], requires_grad=True)
    c = a * b  # c = 6.0
    c.backward()
    
    # dc/da = b = 3.0
    # dc/db = a = 2.0
    assert np.isclose(a.grad.item(), 3.0), f"a.grad = {a.grad}, expected 3.0"
    assert np.isclose(b.grad.item(), 2.0), f"b.grad = {b.grad}, expected 2.0"
    print("✓ Multiplication gradients (product rule) correct")
    print(f"  c = a * b = 2.0 * 3.0")
    print(f"  dc/da = b = {a.grad.item()}")
    print(f"  dc/db = a = {b.grad.item()}")


def test_multiple_appearances():
    """Test: When a tensor appears multiple times in computation"""
    print("\n" + "="*60)
    print("TEST 4: Multiple Appearances (Gradient Accumulation)")
    print("="*60)
    
    a = Tensor([2.0], requires_grad=True)
    # Compute: c = a + a + a = 3*a
    b = a + a
    c = b + a
    c.backward()
    
    # dc/da should be 3.0 (appears 3 times)
    assert np.isclose(a.grad.item(), 3.0), f"a.grad = {a.grad}, expected 3.0"
    print("✓ Gradient accumulation works correctly")
    print(f"  c = a + a + a")
    print(f"  dc/da = 3.0 (from three appearances)")


def test_matrix_multiplication():
    """Test: Matrix multiplication (@)"""
    print("\n" + "="*60)
    print("TEST 5: Matrix Multiplication")
    print("="*60)
    
    # Simple case: (1,2) @ (2,1) = (1,1)
    A = Tensor([[1.0, 2.0]], requires_grad=True)
    B = Tensor([[3.0], [4.0]], requires_grad=True)
    C = A @ B
    
    # Should be: 1*3 + 2*4 = 11
    assert np.isclose(C.data.item(), 11.0), f"C = {C.data}, expected 11.0"
    print(f"✓ Forward pass: [[1,2]] @ [[3],[4]] = [[11]]")
    
    # Backward
    C.backward()
    assert A.grad.shape == (1, 2), f"A.grad shape = {A.grad.shape}"
    assert B.grad.shape == (2, 1), f"B.grad shape = {B.grad.shape}"
    print(f"✓ Backward pass computed correctly")
    print(f"  A.grad.shape = {A.grad.shape}")
    print(f"  B.grad.shape = {B.grad.shape}")


def test_activation_relu():
    """Test: ReLU activation"""
    print("\n" + "="*60)
    print("TEST 6: ReLU Activation")
    print("="*60)
    
    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = x.relu()
    
    expected = [0.0, 0.0, 0.0, 1.0, 2.0]
    assert np.allclose(y.data, expected), f"ReLU output = {y.data}, expected {expected}"
    print(f"✓ ReLU forward: {x.data} → {y.data}")
    
    # Backward
    loss = y.sum()
    loss.backward()
    
    # Gradient should be 1 where x>0, else 0
    expected_grad = [0.0, 0.0, 0.0, 1.0, 1.0]
    assert np.allclose(x.grad, expected_grad), f"ReLU gradient = {x.grad}"
    print(f"✓ ReLU backward: gradient = {x.grad}")


def test_activation_sigmoid():
    """Test: Sigmoid activation"""
    print("\n" + "="*60)
    print("TEST 7: Sigmoid Activation")
    print("="*60)
    
    x = Tensor([0.0], requires_grad=True)
    y = x.sigmoid()
    
    # sigmoid(0) = 0.5
    assert np.isclose(y.data.item(), 0.5), f"sigmoid(0) = {y.data}, expected 0.5"
    print(f"✓ Sigmoid forward: sigmoid(0) = {y.data.item()}")
    
    # Backward
    y.backward()
    # d/dx sigmoid(0) = 0.5 * (1 - 0.5) = 0.25
    assert np.isclose(x.grad.item(), 0.25), f"sigmoid'(0) = {x.grad}, expected 0.25"
    print(f"✓ Sigmoid backward: gradient = {x.grad.item()}")


def test_chain_rule():
    """Test: Complex chain rule through multiple operations"""
    print("\n" + "="*60)
    print("TEST 8: Chain Rule (Complex)")
    print("="*60)
    
    # Compute: loss = ((x * 2 + 3) ** 2).sum()
    x = Tensor([1.0, 2.0], requires_grad=True)
    
    y1 = x * 2           # y1 = [2, 4]
    y2 = y1 + 3          # y2 = [5, 7]
    y3 = y2 ** 2         # y3 = [25, 49]
    loss = y3.sum()      # loss = 74
    
    loss.backward()
    
    # By chain rule: d(loss)/dx = d/dx sum((2x + 3)²)
    #                           = 2*(2x + 3) * 2
    #                           = 4*(2x + 3)
    # At x=[1, 2]: = [4*5, 4*7] = [20, 28]
    expected_grad = [20.0, 28.0]
    assert np.allclose(x.grad, expected_grad), f"gradient = {x.grad}, expected {expected_grad}"
    print(f"✓ Chain rule: d(loss)/dx = {x.grad}")
    print(f"  Computation: loss = sum((2*x + 3)²)")


def test_mse_loss():
    """Test: MSE loss function"""
    print("\n" + "="*60)
    print("TEST 9: MSE Loss")
    print("="*60)
    
    predictions = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    targets = Tensor([[1.0, 1.0], [3.0, 3.0]])
    
    loss = mse_loss(predictions, targets)
    loss.backward()
    
    # MSE = mean((pred - target)²)
    # diff = [[0, 1], [0, 1]]
    # squared = [[0, 1], [0, 1]]
    # MSE = mean([0,1,0,1]) = 0.5
    assert np.isclose(loss.data.item(), 0.5), f"loss = {loss.data}, expected 0.5"
    print(f"✓ MSE loss: {loss.data.item()}")
    
    # Check gradients exist
    assert predictions.grad is not None
    print(f"✓ Gradients computed")
    print(f"  predictions.grad =\n{predictions.grad}")


def test_reshape_and_flatten():
    """Test: Reshape and flatten operations"""
    print("\n" + "="*60)
    print("TEST 10: Reshape & Flatten")
    print("="*60)
    
    x = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    
    # Reshape
    y = x.reshape((2, 3))
    assert y.shape == (2, 3), f"Shape = {y.shape}, expected (2, 3)"
    print(f"✓ Reshape: {x.shape} → {y.shape}")
    
    # Flatten
    z = y.flatten()
    assert z.shape == (6,), f"Shape = {z.shape}, expected (6,)"
    assert np.allclose(z.data, x.data), "Data doesn't match after flatten"
    print(f"✓ Flatten: {y.shape} → {z.shape}")
    
    # Backward through reshape
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    print(f"✓ Backward through reshape works")


def test_reduction_operations():
    """Test: Sum and mean operations"""
    print("\n" + "="*60)
    print("TEST 11: Reduction Operations (Sum & Mean)")
    print("="*60)
    
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    
    # Sum
    sum_all = x.sum()
    assert np.isclose(sum_all.data.item(), 21.0), "Sum failed"
    print(f"✓ Sum all: {sum_all.data.item()}")
    
    # Sum along axis
    x_copy = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    sum_axis0 = x_copy.sum(axis=0)
    assert np.allclose(sum_axis0.data, [5.0, 7.0, 9.0])
    print(f"✓ Sum along axis 0: {sum_axis0.data}")
    
    # Mean
    x_copy2 = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
    mean_all = x_copy2.mean()
    assert np.isclose(mean_all.data.item(), 3.5), "Mean failed"
    print(f"✓ Mean: {mean_all.data.item()}")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("🧪 NEURALFORGE TENSOR TEST SUITE")
    print("="*60)
    
    try:
        test_basic_arithmetic()
        test_gradients_addition()
        test_gradients_multiplication()
        test_multiple_appearances()
        test_matrix_multiplication()
        test_activation_relu()
        test_activation_sigmoid()
        test_chain_rule()
        test_mse_loss()
        test_reshape_and_flatten()
        test_reduction_operations()
        
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour tensor implementation is working correctly!")
        print("Ready to move to Step 2: Neural Network Layers")
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)