"""Loss functions for training neural networks."""

import numpy as np
from neuralforge.core.tensor import Tensor



def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Mean Squared Error loss."""
    diff = predictions - targets
    squared = diff ** 2
    loss = squared.mean()
    return loss


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Cross Entropy Loss for classification."""
    probs = logits.softmax(dim=-1)
    eps = 1e-7
    probs_clamped = Tensor(
        np.clip(probs.data, eps, 1.0),
        requires_grad=False
    )
    log_probs = probs_clamped.log()
    loss = -(targets * log_probs).sum()
    batch_size = targets.shape[0]
    loss = loss / batch_size
    return loss


def nll_loss(log_probs: Tensor, targets: Tensor) -> Tensor:
    """Negative Log Likelihood loss."""
    loss = -(targets * log_probs).sum() / targets.shape[0]
    return loss


def l1_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """L1 (Mean Absolute Error) loss."""
    diff = predictions - targets
    abs_diff = diff.abs()
    loss = abs_diff.mean()
    return loss


class BCELoss:
    """Binary Cross Entropy Loss."""
    
    def __call__(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute binary cross entropy."""
        eps = 1e-7
        preds = Tensor(
            np.clip(predictions.data, eps, 1.0 - eps),
            requires_grad=False
        )
        loss = -(targets * preds.log() + (1 - targets) * (1 - preds).log())
        return loss.mean()

