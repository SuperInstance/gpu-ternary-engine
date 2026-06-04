"""Ecological dynamics for ternary strategy species.

Lotka-Volterra predator-prey dynamics, species interaction matrix,
and equilibrium / stability analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Default interaction matrix (5×5) among species:
#   Explorer, Diplomat, Marksman, Climber, Prospector
#
# Positive α_ij → species i benefits from j.
# Negative α_ij → species i is suppressed by j.
DEFAULT_INTERACTION = np.array(
    [
        #  Exp   Dip   Mar   Cli   Pro
        [0.10, 0.05, -0.20, 0.05, 0.15],  # Explorer
        [0.05, 0.02, 0.10, 0.05, -0.05],  # Diplomat
        [-0.10, 0.15, 0.05, -0.10, 0.20],  # Marksman
        [0.10, 0.05, -0.05, 0.05, 0.10],  # Climber
        [0.05, -0.05, -0.15, 0.10, 0.02],  # Prospector
    ],
    dtype=np.float64,
)


class SpeciesInteractionMatrix:
    """Manages the 5×5 inter-species interaction matrix."""

    def __init__(self, matrix: Optional[np.ndarray] = None) -> None:
        self.matrix = (
            matrix.copy() if matrix is not None else DEFAULT_INTERACTION.copy()
        )
        assert self.matrix.shape == (5, 5), "Interaction matrix must be 5×5"

    def symmetric(self) -> "SpeciesInteractionMatrix":
        """Return a symmetrised copy: (A + A^T) / 2."""
        return SpeciesInteractionMatrix((self.matrix + self.matrix.T) / 2)

    def is_stable(self, growth_rates: np.ndarray, tol: float = 1e-6) -> bool:
        """Check Lyapunov stability at equilibrium (all eigenvalues < 0 real part)."""
        J = self._jacobian(growth_rates)
        eigs = np.linalg.eigvals(J)
        return bool(np.all(eigs.real < tol))

    def _jacobian(self, growth_rates: np.ndarray) -> np.ndarray:
        """Jacobian of Lotka-Volterra at non-trivial equilibrium."""
        r = growth_rates
        A = self.matrix
        # Equilibrium: N* = -r / diag(A) (for negative self-interaction)
        N_star = np.abs(r) / (np.abs(np.diag(A)) + 1e-12)
        J = np.diag(r) + A * N_star[:, None]
        return J


class LotkaVolterra:
    """Discrete-time Lotka-Volterra dynamics for 5 strategy species.

    dN_i/dt = N_i * (r_i + Σ_j α_ij N_j)
    """

    def __init__(
        self,
        growth_rates: Optional[np.ndarray] = None,
        interaction: Optional[SpeciesInteractionMatrix] = None,
        dt: float = 0.01,
        carrying_capacity: float = 1.0,
    ) -> None:
        self.growth_rates = (
            growth_rates if growth_rates is not None else np.array([0.5, 0.3, 0.4, 0.35, 0.25])
        )
        self.interaction = interaction or SpeciesInteractionMatrix()
        self.dt = dt
        self.carrying_capacity = carrying_capacity
        self._history: list[np.ndarray] = []

    @property
    def n_species(self) -> int:
        return len(self.growth_rates)

    def step(self, populations: np.ndarray) -> np.ndarray:
        """Advance one time-step and return new populations."""
        N = populations.copy()
        A = self.interaction.matrix
        r = self.growth_rates

        # Lotka-Volterra with logistic dampening
        dN = N * (r + A @ N) * self.dt
        N_new = N + dN
        # Soft carrying capacity clamp
        N_new = np.clip(N_new, 0, self.carrying_capacity * 2)
        self._history.append(N_new.copy())
        return N_new

    def simulate(
        self, initial: np.ndarray, n_steps: int = 1000
    ) -> np.ndarray:
        """Run *n_steps* and return trajectory (n_steps+1, n_species)."""
        traj = [initial.copy()]
        N = initial.copy()
        for _ in range(n_steps):
            N = self.step(N)
            traj.append(N.copy())
        return np.array(traj)


class StabilityAnalysis:
    """Analyse equilibrium and stability of LV system."""

    def __init__(self, lv: LotkaVolterra) -> None:
        self.lv = lv

    def equilibrium(self, tol: float = 1e-6) -> Optional[np.ndarray]:
        """Numerically find equilibrium by running to convergence."""
        N = np.full(self.lv.n_species, 0.5)
        for _ in range(50_000):
            N_new = self.lv.step(N)
            if np.max(np.abs(N_new - N)) < tol:
                return N_new
            N = N_new
        return N  # best effort

    def eigenvalues(self) -> np.ndarray:
        """Eigenvalues of the Jacobian at approximate equilibrium."""
        N_eq = self.equilibrium()
        assert N_eq is not None
        r = self.lv.growth_rates
        A = self.lv.interaction.matrix
        J = np.diag(r) + A * N_eq[:, None]
        return np.linalg.eigvals(J)

    def is_stable(self) -> bool:
        """True if all eigenvalue real parts are negative."""
        return bool(np.all(self.eigenvalues().real < 0))

    def dominant_timescale(self) -> float:
        """Inverse of the largest (least stable) eigenvalue magnitude."""
        eigs = self.eigenvalues()
        max_real = np.max(np.abs(eigs.real))
        return 1.0 / (max_real + 1e-12)
