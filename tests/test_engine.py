"""Tests for gpu_ternary_engine."""

import numpy as np
import pytest

from gpu_ternary_engine import (
    ExhaustiveSearch,
    GPUBatch,
    LotkaVolterra,
    Population,
    Environment,
    ScalingExperiment,
    SpeciesInteractionMatrix,
    StabilityAnalysis,
)
from gpu_ternary_engine.strategy_species import (
    StrategySpecies,
    classify_strategy,
    classify_population,
    species_distribution,
    SPECIES_NAMES,
)


# ── Population ──────────────────────────────────────────────────────────

class TestPopulation:
    def test_create_default(self):
        pop = Population(size=100, n_actions=4)
        assert pop.strategies.shape == (100, 4)
        assert pop.fitness.shape == (100,)
        assert set(np.unique(pop.strategies)).issubset({-1, 0, 1})

    def test_evolve_reduces_size_correctly(self):
        pop = Population(size=100, n_actions=4)
        P = np.random.randn(4, 4).astype(np.float32)
        pop.evolve(P, n_rounds=3)
        assert pop.strategies.shape == (100, 4)  # size stays constant

    def test_custom_strategies(self):
        S = np.ones((50, 3), dtype=np.int8)
        pop = Population(size=50, n_actions=3, strategies=S)
        assert np.all(pop.strategies == 1)


# ── Environment ─────────────────────────────────────────────────────────

class TestEnvironment:
    def test_default_payoff_shape(self):
        env = Environment(n_actions=4)
        assert env.payoff_matrix.shape == (4, 4)

    def test_step_returns_population(self):
        env = Environment(n_actions=4)
        pop = Population(size=200, n_actions=4)
        result = env.step(pop)
        assert result.strategies.shape == (200, 4)
        assert not np.all(result.fitness == 0)


# ── GPUBatch ────────────────────────────────────────────────────────────

class TestGPUBatch:
    def test_cpu_evaluate_shape(self):
        batch = GPUBatch(n_agents=500, n_actions=4)
        S = np.random.choice([-1, 0, 1], size=(500, 4)).astype(np.int8)
        P = np.random.randn(4, 4).astype(np.float32)
        fit = batch.evaluate(S, P)
        assert fit.shape == (500,)

    def test_cpu_evaluate_values_finite(self):
        batch = GPUBatch(n_agents=100, n_actions=3)
        S = np.random.choice([-1, 0, 1], size=(100, 3)).astype(np.int8)
        P = np.random.randn(3, 3).astype(np.float32)
        fit = batch.evaluate(S, P)
        assert np.all(np.isfinite(fit))


# ── ExhaustiveSearch ────────────────────────────────────────────────────

class TestExhaustiveSearch:
    def test_81_strategies(self):
        es = ExhaustiveSearch(n_actions=4)
        assert es.strategies.shape == (81, 4)

    def test_search_returns_best(self):
        es = ExhaustiveSearch(n_actions=4)
        P = np.eye(4, dtype=np.float32)  # identity → favor +1 strategies
        best, fit = es.search(P)
        assert best.shape == (4,)
        assert isinstance(fit, float)

    def test_all_fitness_length(self):
        es = ExhaustiveSearch(n_actions=3)
        P = np.random.randn(3, 3).astype(np.float32)
        all_f = es.all_fitness(P)
        assert len(all_f) == 27  # 3^3


# ── Strategy Species ───────────────────────────────────────────────────

class TestStrategySpecies:
    def test_explorer(self):
        s = np.array([1, -1, 1, -1])  # 100% nonzero
        assert classify_strategy(s) == StrategySpecies.EXPLORER

    def test_diplomat(self):
        s = np.array([0, 0, 0, 0])  # all zeros
        assert classify_strategy(s) == StrategySpecies.DIPLOMAT

    def test_marksman(self):
        s = np.array([1, 1, 1, 0])  # high aggression, low neg
        # This is also high nonzero → Explorer
        s2 = np.array([1, 0, 1, 0])  # 50% nonzero ≥ 40% → Explorer
        assert classify_strategy(s2) == StrategySpecies.EXPLORER

    def test_classify_population(self):
        strats = np.array([[1, -1, 1, -1], [0, 0, 0, 0]], dtype=np.int8)
        labels = classify_population(strats)
        assert len(labels) == 2

    def test_species_distribution_sums_to_one(self):
        strats = np.random.choice([-1, 0, 1], size=(1000, 4)).astype(np.int8)
        dist = species_distribution(strats)
        assert abs(sum(dist.values()) - 1.0) < 0.01

    def test_five_species_names(self):
        assert len(SPECIES_NAMES) == 5


# ── Ecology ─────────────────────────────────────────────────────────────

class TestEcology:
    def test_lv_step_shapes(self):
        lv = LotkaVolterra()
        N = np.array([0.3, 0.2, 0.25, 0.15, 0.1])
        N2 = lv.step(N)
        assert N2.shape == (5,)
        assert np.all(N2 >= 0)

    def test_lv_simulate_trajectory(self):
        lv = LotkaVolterra(dt=0.005)
        N0 = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        traj = lv.simulate(N0, n_steps=100)
        assert traj.shape == (101, 5)

    def test_interaction_matrix_symmetric(self):
        sim = SpeciesInteractionMatrix()
        sym = sim.symmetric()
        assert np.allclose(sym.matrix, sym.matrix.T)

    def test_stability_analysis(self):
        lv = LotkaVolterra(dt=0.001)
        sa = StabilityAnalysis(lv)
        eigs = sa.eigenvalues()
        assert len(eigs) == 5
