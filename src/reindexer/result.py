"""Intermediate results for graph-based molecular reindexing."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HydrogenPlacement:
    """Coordinate for one hydrogen occupying a reference hydrogen slot."""

    reference_index: int
    target_parent_index: int
    coords: np.ndarray
    geometry: str
    bond_length: float


@dataclass
class ReindexResult:
    """Validated mapping and hydrogen-slot diagnostics.

    Heavy-atom mappings use the original internal graph indices.  Hydrogen
    mappings are kept separately because target hydrogens may be absent.
    """

    target_to_reference: dict[int, int]
    reference_to_target: dict[int, int]
    missing_hydrogens: dict[int, int] = field(default_factory=dict)
    existing_hydrogens: dict[int, list[int]] = field(default_factory=dict)
    ambiguous_mappings: list[dict[int, int]] = field(default_factory=list)
    unmatched_atoms: list[dict[str, Any]] = field(default_factory=list)
    validation: dict[str, Any] = field(default_factory=dict)
    expected_hydrogens: dict[int, int] = field(default_factory=dict)
    reference_hydrogen_to_target: dict[int, int | None] = field(default_factory=dict)
    target_hydrogen_to_reference: dict[int, int] = field(default_factory=dict)
    hydrogen_ambiguities: list[dict[str, Any]] = field(default_factory=list)
    mapping_scores: dict[str, Any] = field(default_factory=dict)
    generated_hydrogens: dict[int, HydrogenPlacement] = field(default_factory=dict)
