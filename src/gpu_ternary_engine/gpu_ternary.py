"""Core GPU/CPU ternary simulation engine.

Provides Population, Environment, GPUBatch, and ExhaustiveSearch for
high-performance ternary agent simulation with graceful CPU fallback.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Optional PyTorch import – graceful CPU fallback when unavailable
# ---------------------------------------------------------------------------
try:
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


def _torch_available() -> bool:
    """Return True if PyTorch *and* CUDA are ready."""
    if not _HAS_TORCH:
        return False
    return torch.cuda.is_available()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Strategy helpers
# ---------------------------------------------------------------------------
TERNARY_VALUES = (-1, 0, 1)


def _enumerate_strategies(n_actions: int = 4) -> np.ndarray:
    """Return an (3^n_actions, n_actions) array of all pure ternary strategies.

    With n_actions=4 this yields 81 strategies – the canonical exhaustive set.
    """
    combos = list(itertools.product(TERNARY_VALUES, repeat=n_actions))
    return np.array(combos, dtype=np.int8)


# ---------------------------------------------------------------------------
# Population
# ---------------------------------------------------------------------------
@dataclass
class Population:
    """A population of agents, each carrying a ternary strategy vector."""

    size: int = 10_000
    n_actions: int = 4
    strategies: Optional[np.ndarray] = None  # (size, n_actions)
    fitness: Optional[np.ndarray] = None  # (size,)
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.strategies is None:
            self.strategies = np.random.choice(
                TERNARY_VALUES, size=(self.size, self.n_actions)
            ).astype(np.int8)
        if self.fitness is None:
            self.fitness = np.zeros(self.size, dtype=np.float32)

    # ------------------------------------------------------------------
    def evolve(self, payoff_matrix: np.ndarray, n_rounds: int = 1) -> None:
        """Evaluate fitness over *n_rounds* and select the top half."""
        for _ in range(n_rounds):
            self._evaluate(payoff_matrix)
            self._select()

    def _evaluate(self, payoff_matrix: np.ndarray) -> None:
        """Compute fitness as strategy @ payoff_matrix @ strategy^T diagonal proxy."""
        self.fitness = (self.strategies.astype(np.float32) @ payoff_matrix).sum(axis=1)

    def _select(self) -> None:
        """Keep top 50 % and replicate with mutation."""
        threshold = np.median(self.fitness)
        survivors = self.strategies[self.fitness >= threshold]
        n_needed = self.size - len(survivors)
        if n_needed <= 0:
            return
        # replicate from survivors with small mutation
        parents = survivors[np.random.randint(0, len(survivors), size=n_needed)]
        mutations = np.random.choice(TERNARY_VALUES, size=parents.shape).astype(np.int8)
        mask = np.random.random(parents.shape) < 0.05  # 5 % mutation rate
        children = np.where(mask, mutations, parents)
        self.strategies = np.vstack([survivors, children]).astype(np.int8)
        self.fitness = np.zeros(self.size, dtype=np.float32)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
@dataclass
class Environment:
    """Simulated environment with a configurable payoff matrix."""

    n_actions: int = 4
    payoff_matrix: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.payoff_matrix is None:
            # Random symmetric-ish payoff in [-1, 1]
            raw = np.random.randn(self.n_actions, self.n_actions).astype(np.float32)
            self.payoff_matrix = (raw + raw.T) / (2 * self.n_actions)

    def step(self, population: Population) -> Population:
        """Run one evaluation step and return the (mutated) population."""
        population._evaluate(self.payoff_matrix)
        return population


# ---------------------------------------------------------------------------
# GPUBatch
# ---------------------------------------------------------------------------
class GPUBatch:
    """Batch evaluator that uses CUDA tensors when available, else NumPy.

    Target throughput:
      - CPU: ≥561 M cells / sec
      - GPU: limited by device memory
    """

    def __init__(self, n_agents: int = 10_000, n_actions: int = 4) -> None:
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.use_cuda = _torch_available()

    # ------------------------------------------------------------------
    def evaluate(
        self,
        strategies: np.ndarray,
        payoff_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute fitness for *strategies* against *payoff_matrix*.

        Returns a 1-D fitness array of length ``strategies.shape[0]``.
        """
        if self.use_cuda:
            return self._evaluate_gpu(strategies, payoff_matrix)
        return self._evaluate_cpu(strategies, payoff_matrix)

    # ------------------------------------------------------------------
    def _evaluate_cpu(
        self, strategies: np.ndarray, payoff_matrix: np.ndarray
    ) -> np.ndarray:
        S = strategies.astype(np.float32)
        P = payoff_matrix.astype(np.float32)
        fitness = S @ P
        fitness = (fitness * S).sum(axis=1)
        return fitness

    def _evaluate_gpu(
        self, strategies: np.ndarray, payoff_matrix: np.ndarray
    ) -> np.ndarray:
        S = torch.tensor(strategies, dtype=torch.float32, device="cuda")
        P = torch.tensor(payoff_matrix, dtype=torch.float32, device="cuda")
        fitness = (S @ P * S).sum(dim=1)
        return fitness.cpu().numpy()


# ---------------------------------------------------------------------------
# ExhaustiveSearch
# ---------------------------------------------------------------------------
class ExhaustiveSearch:
    """Brute-force search over all 3^n_actions pure ternary strategies.

    Default: 81 strategies (n_actions=4).
    """

    def __init__(self, n_actions: int = 4) -> None:
        self.n_actions = n_actions
        self.strategies = _enumerate_strategies(n_actions)
        self.batch = GPUBatch(n_agents=len(self.strategies), n_actions=n_actions)

    def search(self, payoff_matrix: np.ndarray) -> tuple[np.ndarray, float]:
        """Return (best_strategy, best_fitness) from the exhaustive set."""
        fitness = self.batch.evaluate(self.strategies, payoff_matrix)
        idx = int(np.argmax(fitness))
        return self.strategies[idx], float(fitness[idx])

    def all_fitness(self, payoff_matrix: np.ndarray) -> np.ndarray:
        """Return fitness for every strategy."""
        return self.batch.evaluate(self.strategies, payoff_matrix)
