"""Minimal tabular adapters of published open-source DA components.

Core implementations follow THU Transfer-Learning-Library (TLlib):
https://github.com/thuml/Transfer-Learning-Library

Upstream files used as the algorithmic reference:
- tllib/modules/grl.py
- tllib/modules/domain_discriminator.py
- tllib/alignment/dann.py
- tllib/alignment/cdan.py
- tllib/alignment/dan.py
- tllib/modules/kernels.py

TLlib is MIT licensed. This module keeps the same loss definitions and data
flow while removing image-specific classifier wrappers. Deep CORAL follows
https://github.com/VisionLearningGroup/CORAL.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, coefficient: float) -> torch.Tensor:
        ctx.coefficient = coefficient
        return value * 1.0

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return gradient.neg() * ctx.coefficient, None


class WarmStartGradientReverseLayer(nn.Module):
    """TLlib warm-start GRL schedule."""

    def __init__(self, alpha: float = 1.0, lo: float = 0.0, hi: float = 1.0,
                 max_iters: int = 1000, auto_step: bool = True) -> None:
        super().__init__()
        self.alpha, self.lo, self.hi = alpha, lo, hi
        self.max_iters, self.auto_step, self.iter_num = max_iters, auto_step, 0

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        coefficient = (
            2.0 * (self.hi - self.lo)
            / (1.0 + math.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.iter_num += 1
        return GradientReverseFunction.apply(value, coefficient)


class DomainDiscriminator(nn.Sequential):
    """TLlib domain discriminator with sigmoid output."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid(),
        )


def binary_accuracy(probability: torch.Tensor, target: torch.Tensor) -> float:
    return float(((probability >= 0.5) == (target >= 0.5)).float().mean().detach().cpu())


class DomainAdversarialLoss(nn.Module):
    """DANN loss matching TLlib's equal source/target batch definition."""

    def __init__(self, discriminator: nn.Module, max_iters: int) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.grl = WarmStartGradientReverseLayer(max_iters=max_iters, auto_step=True)
        self.domain_discriminator_accuracy = float("nan")

    def forward(self, source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        if source_h.shape[0] != target_h.shape[0]:
            raise ValueError("DANN requires balanced source and target domain batches.")
        domain_probability = self.discriminator(self.grl(torch.cat([source_h, target_h], dim=0)))
        source_probability, target_probability = domain_probability.chunk(2, dim=0)
        source_label = torch.ones_like(source_probability)
        target_label = torch.zeros_like(target_probability)
        self.domain_discriminator_accuracy = 0.5 * (
            binary_accuracy(source_probability, source_label)
            + binary_accuracy(target_probability, target_label)
        )
        return 0.5 * (
            F.binary_cross_entropy(source_probability, source_label)
            + F.binary_cross_entropy(target_probability, target_label)
        )


class MultiLinearMap(nn.Module):
    """Official CDAN outer-product conditioning T(h, g)."""

    def forward(self, feature: torch.Tensor, probability: torch.Tensor) -> torch.Tensor:
        batch_size = feature.shape[0]
        return torch.bmm(probability.unsqueeze(2), feature.unsqueeze(1)).view(batch_size, -1)


class ConditionalDomainAdversarialLoss(nn.Module):
    """CDAN loss with detached softmax predictions, as in TLlib."""

    def __init__(self, discriminator: nn.Module, max_iters: int) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.map = MultiLinearMap()
        self.grl = WarmStartGradientReverseLayer(max_iters=max_iters, auto_step=True)
        self.domain_discriminator_accuracy = float("nan")

    def forward(self, source_logits: torch.Tensor, source_h: torch.Tensor,
                target_logits: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        if source_h.shape[0] != target_h.shape[0]:
            raise ValueError("CDAN requires balanced source and target domain batches.")
        feature = torch.cat([source_h, target_h], dim=0)
        logits = torch.cat([source_logits, target_logits], dim=0)
        probability = F.softmax(logits, dim=1).detach()
        conditioned = self.grl(self.map(feature, probability))
        domain_probability = self.discriminator(conditioned)
        labels = torch.cat([
            torch.ones((source_h.shape[0], 1), device=feature.device),
            torch.zeros((target_h.shape[0], 1), device=feature.device),
        ])
        self.domain_discriminator_accuracy = binary_accuracy(domain_probability, labels)
        return F.binary_cross_entropy(domain_probability, labels)


class GaussianKernel(nn.Module):
    """TLlib adaptive-bandwidth Gaussian kernel."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        squared_distance = ((values.unsqueeze(0) - values.unsqueeze(1)) ** 2).sum(2)
        sigma_square = self.alpha * squared_distance.detach().mean()
        if not bool(torch.isfinite(sigma_square)) or float(sigma_square) <= 0:
            raise ValueError("MK-MMD Gaussian bandwidth is non-positive or non-finite.")
        return torch.exp(-squared_distance / (2.0 * sigma_square))


class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
    """Non-linear MK-MMD estimator from TLlib DAN."""

    def __init__(self, kernels: Sequence[nn.Module]) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(kernels)
        self.index_matrix: Optional[torch.Tensor] = None

    @staticmethod
    def make_index_matrix(batch_size: int) -> torch.Tensor:
        matrix = torch.zeros(2 * batch_size, 2 * batch_size)
        for i in range(batch_size):
            for j in range(batch_size):
                if i != j:
                    matrix[i, j] = 1.0 / (batch_size * (batch_size - 1))
                    matrix[i + batch_size, j + batch_size] = 1.0 / (batch_size * (batch_size - 1))
                matrix[i, j + batch_size] = -1.0 / (batch_size * batch_size)
                matrix[i + batch_size, j] = -1.0 / (batch_size * batch_size)
        return matrix

    def forward(self, source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
        if source_h.shape != target_h.shape or source_h.shape[0] < 2:
            raise ValueError("MK-MMD requires equal source/target batches with at least two samples.")
        batch_size = source_h.shape[0]
        if self.index_matrix is None or self.index_matrix.shape[0] != 2 * batch_size:
            self.index_matrix = self.make_index_matrix(batch_size)
        values = torch.cat([source_h, target_h], dim=0)
        kernel_matrix = sum(kernel(values) for kernel in self.kernels)
        return (kernel_matrix * self.index_matrix.to(values.device)).sum() + 2.0 / (batch_size - 1)


def deep_coral_loss(source_h: torch.Tensor, target_h: torch.Tensor) -> torch.Tensor:
    """Deep CORAL loss: squared covariance distance divided by 4d^2."""

    if source_h.shape[0] < 2 or target_h.shape[0] < 2:
        raise ValueError("Deep CORAL requires at least two samples from each domain.")

    def covariance(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=0, keepdim=True)
        return centered.T @ centered / (values.shape[0] - 1)

    dimension = source_h.shape[1]
    return (covariance(source_h) - covariance(target_h)).pow(2).sum() / (4.0 * dimension * dimension)


def class_centroid_loss(source_h: torch.Tensor, source_y: torch.Tensor,
                        target_h: torch.Tensor, target_logits: torch.Tensor,
                        classes: int) -> torch.Tensor:
    """Project extension: soft pseudo-label class-wise latent centroid matching."""

    target_weight = F.softmax(target_logits, dim=1).detach()
    losses = []
    for class_index in range(classes):
        source_mask = source_y == class_index
        if not bool(source_mask.any()):
            continue
        source_center = source_h[source_mask].mean(dim=0)
        weight = target_weight[:, class_index]
        if float(weight.sum()) <= 1e-9:
            continue
        target_center = (target_h * weight.unsqueeze(1)).sum(dim=0) / weight.sum()
        losses.append((source_center - target_center).pow(2).mean())
    if not losses:
        raise ValueError("No class centroid could be computed for the current batch.")
    return torch.stack(losses).mean()
