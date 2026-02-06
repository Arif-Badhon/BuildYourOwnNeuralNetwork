# NeuralForge 🧠🔨

**Build your own Neural Network library from scratch.**

NeuralForge is an educational deep learning framework designed to demystify the internal mechanics of modern libraries like PyTorch. It implements a custom **autograd engine** and a modular neural network API, allowing you to build, train, and understand neural networks from first principles.

> **Note**: This project is for educational purposes. For production usage, please use PyTorch or TensorFlow.

---

## 🚀 Key Features

- **Automatic Differentiation (Autograd)**: A custom Tensor engine that tracks the computation graph and calculates gradients automatically via reverse-mode AD.
- **PyTorch-like API**: Familiar syntax and structure (`Module`, `Optimizer`, `backward()`) to ease the learning curve.
- **Modular Design**: clear separation between the tensor core (`src/neuralforge/core`), neural network modules (`src/neuralforge/nn`), and optimization algorithms (`src/neuralforge/optim`).
- **Numpy-based**: Built on top of NumPy for efficient numerical computation.

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup via `uv` (Recommended)

```bash
# Clone the repository
git clone https://github.com/Arif-Badhon/neuralforge.git
cd neuralforge

# Create a virtual environment and install dependencies
uv sync
```

### Setup via `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Docker Setup

You can also run the project using Docker:

```bash
docker-compose up --build
```

---

## 💡 Usage Examples

### 1. Basic Tensor Operations & Autograd

The `Tensor` class is the heart of NeuralForge. It behaves like a NumPy array but remembers its history.

```python
from neuralforge.core.tensor import Tensor

# Create tensors with gradient tracking enabled
a = Tensor([2.0, 3.0], requires_grad=True)
b = Tensor([4.0, 5.0], requires_grad=True)

# Perform operations
c = a * b           # Element-wise multiplication
d = c.sum()         # Reduction

# Compute gradients (backpropagation)
d.backward()

print(f"a.grad: {a.grad}")  # Should match b.data
print(f"b.grad: {b.grad}")  # Should match a.data
```

### 2. Recurrent Neural Networks (RNN)

NeuralForge includes a vanilla `RNN` module to demonstrate sequence modeling.

```python
from neuralforge.nn.rnn import RNN
from neuralforge.core import Tensor

# Input: Sequence of Tensors (batch_size, input_size)
x1 = Tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
x2 = Tensor([[0.0, 1.0], [1.0, 0.0]], requires_grad=True)
inputs = [x1, x2]

# Initialize RNN
rnn = RNN(input_size=2, hidden_size=4)

# Forward pass (Unrolling loop)
hidden_states, final_hidden = rnn(inputs)

# Backpropagate through time
final_hidden.sum().backward()
```

### 3. Building a Simple Neural Network (Planned)

*Note: The `nn` module is currently under active development. Below is a preview of the API design.*

```python
# (Concept Code)
from neuralforge.nn import Linear, Module
from neuralforge.core import Tensor

class SimpleNet(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

model = SimpleNet()
output = model(Tensor([1.0, ...]))
output.backward()
```

---

## 📂 Project Structure

```text
neuralforge/
├── src/neuralforge/      # Main library package
│   ├── core/             # Core engine (Tensor, Autograd)
│   ├── nn/               # Neural Network layers (Linear, Conv2d, etc.)
│   ├── optim/            # Optimizers (SGD, Adam)
│   ├── data/             # Data loading utilities
│   └── api/              # API for serving models
├── scripts/              # Training and utility scripts
├── config/               # Configuration files
├── tests/                # Unit tests
└── docker-compose.yml    # Docker services config
```

## 🧪 Development & Testing

To run the test suite:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src
```

## 🤝 Contributing

Contributions are welcome! This is an open learning project.
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `pyproject.toml` for details.
