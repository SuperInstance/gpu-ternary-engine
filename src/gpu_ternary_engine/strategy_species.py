"""Strategy species classification.

Five ecological niches for ternary strategies:
  Explorer, Diplomat, Marksman, Climber, Prospector.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import numpy as np


class StrategySpecies(IntEnum):
    """Canonical species identifiers."""

    EXPLORER = 0
    DIPLOMAT = 1
    MARKSMAN = 2
    CLIMBER = 3
    PROSPECTOR = 4


SPECIES_NAMES: dict[StrategySpecies, str] = {
    StrategySpecies.EXPLORER: "Explorer",
    StrategySpecies.DIPLOMAT: "Diplomat",
    StrategySpecies.MARKSMAN: "Marksman",
    StrategySpecies.CLIMBER: "Climber",
    StrategySpecies.PROSPECTOR: "Prospector",
}

# Classification thresholds ------------------------------------------------
_AGGRESSION_THRESHOLD = 0.3
_PASSIVITY_THRESHOLD = 0.3
_BALANCE_WINDOW = 0.15
_EXPLORE_RATIO_MIN = 0.4
_OPPORTUNISM_THRESHOLD = 0.35


def classify_strategy(strategy: np.ndarray) -> StrategySpecies:
    """Classify a single ternary strategy vector into a species.

    Decision logic (evaluated in order, first match wins):

    1. **Explorer** – high fraction of non-zero (|1|) actions; seeks novelty.
    2. **Diplomat** – low aggression (few +1) and low conflict (few −1); avoids
       extremes.
    3. **Marksman** – high aggression, low passivity; focused attacker.
    4. **Climber** – balanced mix with a slight positive bias; steady improver.
    5. **Prospector** – high passivity, waits for opportunities.
    """
    n = len(strategy)
    if n == 0:
        return StrategySpecies.DIPLOMAT

    pos = np.sum(strategy == 1) / n
    neg = np.sum(strategy == -1) / n
    zero = np.sum(strategy == 0) / n
    nonzero = 1.0 - zero

    # Explorer: lots of non-zero moves
    if nonzero >= _EXPLORE_RATIO_MIN:
        return StrategySpecies.EXPLORER

    # Diplomat: avoids extremes
    if pos < _PASSIVITY_THRESHOLD and neg < _PASSIVITY_THRESHOLD:
        return StrategySpecies.DIPLOMAT

    # Marksman: aggressive, not passive
    if pos >= _AGGRESSION_THRESHOLD and neg < _PASSIVITY_THRESHOLD:
        return StrategySpecies.MARKSMAN

    # Prospector: high passivity, waits
    if zero >= _OPPORTUNISM_THRESHOLD:
        return StrategySpecies.PROSPECTOR

    # Climber: everything else (balanced with slight positive bias)
    return StrategySpecies.CLIMBER


def classify_population(strategies: np.ndarray) -> np.ndarray:
    """Vectorised species classification for a population.

    Returns an int array of species ids with shape (strategies.shape[0],).
    """
    n = strategies.shape[1]
    pos = (strategies == 1).sum(axis=1) / n
    neg = (strategies == -1).sum(axis=1) / n
    zero = (strategies == 0).sum(axis=1) / n
    nonzero = 1.0 - zero

    species = np.full(strategies.shape[0], StrategySpecies.CLIMBER, dtype=np.int8)

    # Apply in reverse priority so higher-priority rules overwrite
    species[zero >= _OPPORTUNISM_THRESHOLD] = StrategySpecies.PROSPECTOR
    species[(pos >= _AGGRESSION_THRESHOLD) & (neg < _PASSIVITY_THRESHOLD)] = (
        StrategySpecies.MARKSMAN
    )
    species[(pos < _PASSIVITY_THRESHOLD) & (neg < _PASSIVITY_THRESHOLD)] = (
        StrategySpecies.DIPLOMAT
    )
    species[nonzero >= _EXPLORE_RATIO_MIN] = StrategySpecies.EXPLORER

    return species


def species_distribution(strategies: np.ndarray) -> dict[str, float]:
    """Return fraction of population in each species."""
    labels = classify_population(strategies)
    dist: dict[str, float] = {}
    for sp in StrategySpecies:
        dist[SPECIES_NAMES[sp]] = float((labels == sp.value).mean())
    return dist
