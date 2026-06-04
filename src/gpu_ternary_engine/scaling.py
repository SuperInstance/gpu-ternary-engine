"""Scaling experiments for ternary agent simulation.

Runs populations at 24 → 240 → 2 400 → 24 000 agents and measures
throughput, correctness, and species distribution convergence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ecology import LotkaVolterra, SpeciesInteractionMatrix
from .gpu_ternary import Environment, ExhaustiveSearch, GPUBatch, Population
from .strategy_species import classify_population, species_distribution


@dataclass
class ScaleResult:
    """Results from a single scale run."""

    n_agents: int
    elapsed_sec: float
    cells_per_sec: float
    species_dist: dict[str, float]
    best_fitness: float


class ScalingExperiment:
    """Run the full 24 → 240 → 2 400 → 24 000 scaling suite."""

    SCALES = [24, 240, 2_400, 24_000]

    def __init__(self, n_actions: int = 4, n_rounds: int = 5) -> None:
        self.n_actions = n_actions
        self.n_rounds = n_rounds

    def run_one(self, n_agents: int) -> ScaleResult:
        """Run a single scale benchmark."""
        env = Environment(n_actions=self.n_actions)
        pop = Population(size=n_agents, n_actions=self.n_actions)
        batch = GPUBatch(n_agents=n_agents, n_actions=self.n_actions)

        total_cells = 0
        t0 = time.perf_counter()

        for _ in range(self.n_rounds):
            fitness = batch.evaluate(pop.strategies, env.payoff_matrix)
            pop.fitness = fitness
            pop._select()
            total_cells += n_agents * self.n_actions

        elapsed = time.perf_counter() - t0
        cells_per_sec = total_cells / max(elapsed, 1e-12)

        dist = species_distribution(pop.strategies)
        best = float(pop.fitness.max())

        return ScaleResult(
            n_agents=n_agents,
            elapsed_sec=elapsed,
            cells_per_sec=cells_per_sec,
            species_dist=dist,
            best_fitness=best,
        )

    def run_all(self) -> list[ScaleResult]:
        """Run the full scaling suite."""
        results: list[ScaleResult] = []
        for n in self.SCALES:
            results.append(self.run_one(n))
        return results

    def exhaustive_at_scale(self, n_actions: int = 4) -> dict[str, Any]:
        """Run exhaustive search and return best strategy + fitness."""
        searcher = ExhaustiveSearch(n_actions=n_actions)
        env = Environment(n_actions=n_actions)
        best_strat, best_fit = searcher.search(env.payoff_matrix)
        all_fit = searcher.all_fitness(env.payoff_matrix)
        return {
            "n_strategies": len(searcher.strategies),
            "best_strategy": best_strat.tolist(),
            "best_fitness": best_fit,
            "mean_fitness": float(all_fit.mean()),
            "fitness_std": float(all_fit.std()),
        }
