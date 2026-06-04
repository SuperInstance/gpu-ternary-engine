"""GPU-accelerated backend for ternary agent simulation."""

__version__ = "0.1.0"

from .gpu_ternary import Population, Environment, GPUBatch, ExhaustiveSearch
from .strategy_species import StrategySpecies, classify_strategy, SPECIES_NAMES
from .ecology import LotkaVolterra, SpeciesInteractionMatrix, StabilityAnalysis
from .scaling import ScalingExperiment

__all__ = [
    "Population",
    "Environment",
    "GPUBatch",
    "ExhaustiveSearch",
    "StrategySpecies",
    "classify_strategy",
    "SPECIES_NAMES",
    "LotkaVolterra",
    "SpeciesInteractionMatrix",
    "StabilityAnalysis",
    "ScalingExperiment",
]
