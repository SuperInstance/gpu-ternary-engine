# gpu-ternary-engine

GPU-accelerated backend for ternary agent simulation.

## Overview

`gpu-ternary-engine` provides high-performance simulation of agents using **ternary strategies** (actions ∈ {-1, 0, +1}). It supports CPU-only execution at 561M+ cells/sec and optional CUDA acceleration via PyTorch.

### Key Components

| Module | Description |
|---|---|
| `gpu_ternary` | Core engine: `Population`, `Environment`, `GPUBatch`, `ExhaustiveSearch` |
| `strategy_species` | 5-species classification: Explorer, Diplomat, Marksman, Climber, Prospector |
| `ecology` | Lotka-Volterra dynamics, species interaction matrix, stability analysis |
| `scaling` | 24→240→2400→24000 agent scaling experiments |

## Installation

```bash
pip install gpu-ternary-engine

# Optional GPU support
pip install gpu-ternary-engine[gpu]
```

## Quick Start

```python
from gpu_ternary_engine import Population, Environment, ExhaustiveSearch
from gpu_ternary_engine.strategy_species import species_distribution

# Create environment and population
env = Environment(n_actions=4)
pop = Population(size=10_000, n_actions=4)

# Evolve for several rounds
pop.evolve(env.payoff_matrix, n_rounds=10)

# Exhaustive search over all 81 strategies
searcher = ExhaustiveSearch(n_actions=4)
best_strat, best_fit = searcher.search(env.payoff_matrix)
print(f"Best strategy: {best_strat}, fitness: {best_fit:.4f}")

# Species distribution
dist = species_distribution(pop.strategies)
for species, fraction in dist.items():
    print(f"  {species}: {fraction:.1%}")
```

## Species Ecology

Each ternary strategy falls into one of five ecological niches:

| Species | Description | Signature |
|---|---|---|
| 🧭 **Explorer** | High action rate, seeks novelty | ≥40% non-zero actions |
| 🤝 **Diplomat** | Avoids extremes, cooperative | Low aggression & conflict |
| 🎯 **Marksman** | Focused aggression, low passivity | High +1, low −1 |
| 🧗 **Climber** | Balanced with positive bias | Default / mixed |
| ⛏️ **Prospector** | Patient, waits for opportunity | ≥35% zeros (passive) |

Species interact via a 5×5 Lotka-Volterra interaction matrix. The system supports equilibrium analysis and eigenvalue-based stability checks.

## Scaling Benchmarks

The scaling suite runs populations at 24, 240, 2,400, and 24,000 agents:

```python
from gpu_ternary_engine import ScalingExperiment

exp = ScalingExperiment(n_actions=4, n_rounds=5)
results = exp.run_all()
for r in results:
    print(f"  {r.n_agents:>6,} agents: {r.cells_per_sec/1e6:.0f}M cells/sec")
```

### Expected Throughput (CPU)

| Agents | Target |
|---|---|
| 24 | Fast warmup |
| 240 | ~500M cells/sec |
| 2,400 | ~560M cells/sec |
| 24,000 | 560M+ cells/sec |

### Exhaustive Search

With `n_actions=4`, all **81 pure ternary strategies** are enumerated and evaluated in a single batch. Best-strategy search completes in <1ms for 10K agents.

## Architecture

```
                    ┌──────────────┐
                    │  Environment │
                    │ (payoff mx)  │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────────┐
        │Population│ │ GPUBatch │ │ ExhaustiveS. │
        │  10K ag. │ │ CPU/CUDA │ │ 81 strategies│
        └────┬─────┘ └────┬─────┘ └──────┬───────┘
             │            │               │
             ▼            ▼               ▼
        ┌─────────────────────────────────────┐
        │         strategy_species            │
        │  Explorer Diplomat Marksman ...     │
        └─────────────────┬───────────────────┘
                          ▼
               ┌────────────────────┐
               │ Lotka-Volterra     │
               │ ecology & stability│
               └────────────────────┘
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT © SuperInstance
