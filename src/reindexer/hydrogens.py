"""Reference-authoritative hydrogen counting and slot assignment."""

from dataclasses import replace
from collections import Counter
from itertools import permutations
from typing import Any

import networkx as nx
import numpy as np

from .result import HydrogenPlacement, ReindexResult


class HydrogenMappingError(ValueError):
    """Raised when target hydrogen topology cannot satisfy the reference."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"Hydrogen mapping failed: {diagnostics}")


class HydrogenGenerationError(ValueError):
    """Raised when target geometry cannot support deterministic placement."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"Hydrogen generation failed: {diagnostics}")


def _hydrogen_name(graph: nx.Graph, node: int) -> str | None:
    value = graph.nodes[node].get("atom_name")
    return str(value).strip() if value is not None else None


def analyse_hydrogens(
    reference: nx.Graph,
    target: nx.Graph,
    result: ReindexResult,
) -> ReindexResult:
    """Assign target hydrogens to reference slots and omit surplus atoms."""
    expected: dict[int, int] = {target_node: 0 for target_node in result.target_to_reference}
    reference_slots: dict[int, int | None] = {}
    reference_hydrogens_by_parent: dict[int, list[int]] = {}
    for reference_h in sorted(
        node for node, data in reference.nodes(data=True) if data["element"] == "H"
    ):
        heavy_neighbors = [
            node for node in reference.neighbors(reference_h) if reference.nodes[node]["element"] != "H"
        ]
        if len(heavy_neighbors) != 1:
            raise HydrogenMappingError(
                {"reference_hydrogen": reference_h, "heavy_neighbors": heavy_neighbors}
            )
        reference_parent = heavy_neighbors[0]
        target_parent = result.reference_to_target[reference_parent]
        expected[target_parent] += 1
        reference_slots[reference_h] = None
        reference_hydrogens_by_parent.setdefault(target_parent, []).append(reference_h)

    target_hydrogens_by_parent: dict[int, list[int]] = {}
    unattached: list[int] = []
    multiply_attached: list[dict[str, Any]] = []
    for target_h in sorted(
        node for node, data in target.nodes(data=True) if data["element"] == "H"
    ):
        heavy_neighbors = [
            node for node in target.neighbors(target_h) if target.nodes[node]["element"] != "H"
        ]
        if len(heavy_neighbors) == 0:
            unattached.append(target_h)
        elif len(heavy_neighbors) != 1:
            multiply_attached.append(
                {"target_hydrogen": target_h, "heavy_neighbors": heavy_neighbors}
            )
        else:
            target_hydrogens_by_parent.setdefault(heavy_neighbors[0], []).append(target_h)

    excess = {
        target_parent: {
            "expected": expected[target_parent],
            "existing": len(target_hydrogens_by_parent.get(target_parent, [])),
        }
        for target_parent in expected
        if len(target_hydrogens_by_parent.get(target_parent, [])) > expected[target_parent]
    }
    diagnostics = {
        "unattached_target_hydrogens": unattached,
        "multiply_attached_target_hydrogens": multiply_attached,
        "excess_hydrogens": excess,
    }
    if unattached or multiply_attached:
        raise HydrogenMappingError(diagnostics)

    target_to_reference_hydrogen: dict[int, int] = {}
    hydrogen_ambiguities: list[dict[str, Any]] = []
    retained_target_hydrogens_by_parent: dict[int, list[int]] = {
        target_parent: [] for target_parent in expected
    }
    for target_parent, reference_slots_for_parent in reference_hydrogens_by_parent.items():
        target_hydrogens = list(target_hydrogens_by_parent.get(target_parent, []))
        remaining_reference = list(reference_slots_for_parent)
        remaining_target = list(target_hydrogens)

        reference_names = [
            _hydrogen_name(reference, ref_h)
            for ref_h in remaining_reference
            if _hydrogen_name(reference, ref_h)
        ]
        target_names = [
            _hydrogen_name(target, target_h)
            for target_h in remaining_target
            if _hydrogen_name(target, target_h)
        ]
        reference_by_name = {
            _hydrogen_name(reference, ref_h): ref_h
            for ref_h in remaining_reference
            if _hydrogen_name(reference, ref_h)
        }
        target_by_name = {
            _hydrogen_name(target, target_h): target_h
            for target_h in remaining_target
            if _hydrogen_name(target, target_h)
        }
        unique_reference_names = {
            name for name, count in Counter(reference_names).items() if count == 1
        }
        unique_target_names = {
            name for name, count in Counter(target_names).items() if count == 1
        }
        for name in sorted(unique_reference_names & unique_target_names):
            ref_h = reference_by_name[name]
            target_h = target_by_name[name]
            reference_slots[ref_h] = target_h
            target_to_reference_hydrogen[target_h] = ref_h
            remaining_reference.remove(ref_h)
            remaining_target.remove(target_h)

        if len(remaining_reference) > 1 and remaining_target:
            hydrogen_ambiguities.append(
                {
                    "target_parent": target_parent,
                    "reference_slots": list(remaining_reference),
                    "target_hydrogens": list(remaining_target),
                    "reason": "chemically equivalent hydrogen slots assigned deterministically",
                }
            )
        retained_remaining_target = remaining_target[: len(remaining_reference)]
        for ref_h, target_h in zip(remaining_reference, retained_remaining_target):
            reference_slots[ref_h] = target_h
            target_to_reference_hydrogen[target_h] = ref_h
        retained_target_hydrogens_by_parent[target_parent] = sorted(
            target_to_reference_hydrogen.keys()
            & set(target_hydrogens_by_parent.get(target_parent, []))
        )

    retained_target_hydrogens = set(target_to_reference_hydrogen)
    omitted_target_hydrogens = sorted(
        target_hydrogen
        for target_hydrogens in target_hydrogens_by_parent.values()
        for target_hydrogen in target_hydrogens
        if target_hydrogen not in retained_target_hydrogens
    )

    missing = {
        target_parent: expected[target_parent]
        - len(retained_target_hydrogens_by_parent.get(target_parent, []))
        for target_parent in expected
    }
    validation = dict(result.validation)
    validation.update(
        {
            "reference_hydrogen_count": sum(expected.values()),
            "target_hydrogen_count": sum(
                len(nodes) for nodes in retained_target_hydrogens_by_parent.values()
            ),
            "omitted_target_hydrogen_count": len(omitted_target_hydrogens),
            "omitted_target_hydrogens": omitted_target_hydrogens,
            "expected_hydrogens_match_reference": True,
            "hydrogen_slots_complete": all(
                target_h is not None for target_h in reference_slots.values()
            ),
            "unattached_target_hydrogens": unattached,
            "hydrogen_ambiguity_count": len(hydrogen_ambiguities),
        }
    )
    return replace(
        result,
        missing_hydrogens=missing,
        existing_hydrogens={
            parent: list(retained_target_hydrogens_by_parent.get(parent, []))
            for parent in expected
        },
        expected_hydrogens=expected,
        reference_hydrogen_to_target=reference_slots,
        target_hydrogen_to_reference=target_to_reference_hydrogen,
        hydrogen_ambiguities=hydrogen_ambiguities,
        validation=validation,
    )


# Bond lengths are deliberately a small geometry table, not a valence model.
HYDROGEN_BOND_LENGTHS = {
    "C": 1.09,
    "N": 1.01,
    "O": 0.96,
    "P": 1.42,
    "S": 1.34,
}


def _unit(vector: np.ndarray, description: str) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise HydrogenGenerationError({"reason": "zero-length vector", "description": description})
    return vector / norm


def _least_parallel_axis(vector: np.ndarray) -> np.ndarray:
    vector = _unit(vector, "fallback axis")
    axes = np.eye(3)
    return axes[int(np.argmin(np.abs(axes @ vector)))]


def _rotation_align_one(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a deterministic row-vector rotation mapping source onto target."""
    source = _unit(source, "template vector")
    target = _unit(target, "observed neighbor vector")
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = _unit(np.cross(source, _least_parallel_axis(source)), "opposite-vector axis")
        return -np.eye(3) + 2.0 * np.outer(axis, axis)

    axis_matrix = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    column_rotation = (
        np.eye(3)
        + axis_matrix
        + axis_matrix @ axis_matrix * ((1.0 - cosine) / (sine * sine))
    )
    return column_rotation.T


def _fit_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if len(source) == 0:
        return np.eye(3)
    if len(source) == 1:
        return _rotation_align_one(source[0], target[0])

    covariance = np.asarray(source, dtype=float).T @ np.asarray(target, dtype=float)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt)) or 1.0
    return u @ correction @ vt


def _fit_template(
    template: np.ndarray,
    observed: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Fit a fixed template to observed heavy-neighbor directions."""
    observed = np.asarray(observed, dtype=float)
    if len(observed) == 0:
        return template.copy(), ()

    best_error = float("inf")
    best_key: tuple[int, ...] | None = None
    best_rotation = np.eye(3)
    for key in permutations(range(len(template)), len(observed)):
        source = template[list(key)]
        rotation = _fit_rotation(source, observed)
        fitted = source @ rotation
        error = float(np.sum(1.0 - np.clip(np.sum(fitted * observed, axis=1), -1.0, 1.0)))
        if error < best_error - 1e-10 or (
            np.isclose(error, best_error, atol=1e-10, rtol=1e-10)
            and (best_key is None or key < best_key)
        ):
            best_error = error
            best_key = key
            best_rotation = rotation
    return template @ best_rotation, best_key or ()


def _tetrahedral_template() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
        ],
        dtype=float,
    ) / np.sqrt(3.0)


def _planar_template() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-0.5, np.sqrt(3.0) / 2.0, 0.0],
            [-0.5, -np.sqrt(3.0) / 2.0, 0.0],
        ],
        dtype=float,
    )


def _pyramidal_template() -> np.ndarray:
    bond_angle = np.deg2rad(107.0)
    axial = np.sqrt((np.cos(bond_angle) + 0.5) / 1.5)
    radial = np.sqrt(1.0 - axial * axial)
    return np.asarray(
        [
            [radial, 0.0, axial],
            [-radial / 2.0, radial * np.sqrt(3.0) / 2.0, axial],
            [-radial / 2.0, -radial * np.sqrt(3.0) / 2.0, axial],
        ],
        dtype=float,
    )


def _heavy_angle(observed: np.ndarray) -> float | None:
    if len(observed) != 2:
        return None
    cosine = np.clip(np.dot(observed[0], observed[1]), -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _geometry_kind(element: str, observed: np.ndarray, total_hydrogens: int) -> str:
    heavy_degree = len(observed)
    total_coordination = heavy_degree + total_hydrogens
    if element == "O" and heavy_degree == 1 and total_hydrogens == 1:
        return "terminal"
    if heavy_degree == 1 and total_hydrogens == 1:
        return "terminal"
    if total_coordination == 4:
        return "tetrahedral"
    if element == "N" and total_coordination == 3:
        angle = _heavy_angle(observed)
        if angle is not None and angle >= 114.0:
            return "trigonal_planar"
        return "trigonal_pyramidal"
    if total_coordination == 3:
        return "trigonal_planar"
    if total_coordination == 2 and heavy_degree >= 1:
        return "linear"
    raise HydrogenGenerationError(
        {
            "reason": "unsupported coordination for deterministic placement",
            "element": element,
            "heavy_degree": heavy_degree,
            "hydrogen_count": total_hydrogens,
        }
    )


def _directions_for_atom(
    element: str,
    parent_coords: np.ndarray,
    neighbor_coords: np.ndarray,
    total_hydrogens: int,
) -> tuple[str, list[np.ndarray]]:
    observed = np.asarray(
        [_unit(coords - parent_coords, "heavy-neighbor vector") for coords in neighbor_coords]
    )
    kind = _geometry_kind(element, observed, total_hydrogens)

    if kind == "terminal":
        return kind, [_unit(-np.sum(observed, axis=0), "terminal hydrogen direction")]

    if kind == "linear":
        template = np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    elif kind == "tetrahedral":
        template = _tetrahedral_template()
    elif kind == "trigonal_planar":
        template = _planar_template()
    else:
        template = _pyramidal_template()

    fitted, occupied = _fit_template(template, observed)
    directions = [fitted[index] for index in range(len(fitted)) if index not in occupied]
    if len(directions) != total_hydrogens:
        raise HydrogenGenerationError(
            {
                "reason": "template coordination mismatch",
                "geometry": kind,
                "heavy_degree": len(observed),
                "hydrogen_count": total_hydrogens,
                "available_directions": len(directions),
            }
        )
    return kind, [_unit(direction, "template hydrogen direction") for direction in directions]


def _reference_hydrogen_slots(
    reference: nx.Graph,
    result: ReindexResult,
) -> dict[int, list[int]]:
    slots: dict[int, list[int]] = {}
    for reference_hydrogen, target_hydrogen in result.reference_hydrogen_to_target.items():
        heavy_neighbors = [
            node for node in reference.neighbors(reference_hydrogen)
            if reference.nodes[node]["element"] != "H"
        ]
        if len(heavy_neighbors) != 1:
            raise HydrogenGenerationError(
                {
                    "reason": "reference hydrogen does not have one heavy parent",
                    "reference_hydrogen": reference_hydrogen,
                    "heavy_neighbors": heavy_neighbors,
                }
            )
        target_parent = result.reference_to_target[heavy_neighbors[0]]
        slots.setdefault(target_parent, []).append(reference_hydrogen)
    for target_parent in slots:
        slots[target_parent].sort()
    return slots


def generate_missing_hydrogens(
    reference: nx.Graph,
    target: nx.Graph,
    result: ReindexResult,
) -> ReindexResult:
    """Generate missing hydrogen coordinates from target heavy geometry only.

    The returned placements are keyed by reference hydrogen index.  Existing
    target hydrogen coordinates are never read or modified.
    """
    slots_by_parent = _reference_hydrogen_slots(reference, result)
    generated: dict[int, HydrogenPlacement] = {}
    diagnostics: dict[str, Any] = {
        "generated_hydrogen_count": 0,
        "geometry_by_parent": {},
        "generated_reference_indices": [],
    }

    for target_parent, missing_count in sorted(result.missing_hydrogens.items()):
        if missing_count == 0:
            continue
        slots = slots_by_parent.get(target_parent, [])
        missing_slots = [
            reference_hydrogen
            for reference_hydrogen in slots
            if result.reference_hydrogen_to_target.get(reference_hydrogen) is None
        ]
        if len(missing_slots) != missing_count:
            raise HydrogenGenerationError(
                {
                    "reason": "missing hydrogen count does not match reference slots",
                    "target_parent": target_parent,
                    "missing_count": missing_count,
                    "missing_slots": missing_slots,
                }
            )

        parent_coords = np.asarray(target.nodes[target_parent]["coords"], dtype=float).copy()
        heavy_neighbors = [
            node for node in target.neighbors(target_parent)
            if target.nodes[node]["element"] != "H"
        ]
        neighbor_coords = np.asarray(
            [target.nodes[node]["coords"] for node in heavy_neighbors], dtype=float
        )
        element = target.nodes[target_parent]["element"]
        total_hydrogens = result.expected_hydrogens[target_parent]
        try:
            geometry, directions = _directions_for_atom(
                element,
                parent_coords,
                neighbor_coords,
                total_hydrogens,
            )
        except HydrogenGenerationError as exc:
            details = dict(getattr(exc, "diagnostics", {}) or {})
            details.update(
                {
                    "target_parent": target_parent,
                    "target_parent_atom": dict(target.nodes[target_parent]),
                    "reference_parent": next(
                        (
                            reference_node
                            for reference_node, target_node
                            in result.reference_to_target.items()
                            if target_node == target_parent
                        ),
                        None,
                    ),
                    "target_heavy_neighbors": [
                        {
                            "index": neighbor,
                            "element": target.nodes[neighbor].get("element"),
                            "atom_name": target.nodes[neighbor].get("atom_name"),
                            "original_index": target.nodes[neighbor].get("original_index"),
                        }
                        for neighbor in heavy_neighbors
                    ],
                    "target_graph_nodes": [
                        {
                            "index": node,
                            "element": data.get("element"),
                            "atom_name": data.get("atom_name"),
                            "original_index": data.get("original_index"),
                        }
                        for node, data in target.nodes(data=True)
                    ],
                    "target_to_reference_mapping": dict(result.target_to_reference),
                    "reference_to_target_mapping": dict(result.reference_to_target),
                }
            )
            raise HydrogenGenerationError(details) from exc
        bond_length = HYDROGEN_BOND_LENGTHS.get(element, 1.00)
        diagnostics["geometry_by_parent"][target_parent] = geometry

        # Directions correspond to all reference slots; only missing slots are emitted.
        for reference_hydrogen, direction in zip(slots, directions):
            if result.reference_hydrogen_to_target[reference_hydrogen] is not None:
                continue
            coords = parent_coords + bond_length * direction
            if not np.isfinite(coords).all():
                raise HydrogenGenerationError(
                    {"reason": "generated coordinates are not finite", "reference_hydrogen": reference_hydrogen}
                )
            generated[reference_hydrogen] = HydrogenPlacement(
                reference_index=reference_hydrogen,
                target_parent_index=target_parent,
                coords=coords,
                geometry=geometry,
                bond_length=bond_length,
            )

    diagnostics["generated_hydrogen_count"] = len(generated)
    diagnostics["generated_reference_indices"] = sorted(generated)
    if len(generated) != sum(result.missing_hydrogens.values()):
        raise HydrogenGenerationError(
            {
                "reason": "generated hydrogen count mismatch",
                "expected": sum(result.missing_hydrogens.values()),
                "generated": len(generated),
            }
        )

    validation = dict(result.validation)
    validation.update(
        {
            "generated_hydrogen_count": len(generated),
            "generated_hydrogen_slots_complete": all(
                reference_hydrogen in generated
                or target_hydrogen is not None
                for reference_hydrogen, target_hydrogen in result.reference_hydrogen_to_target.items()
            ),
            "hydrogen_generation_uses_reference_coordinates": False,
            "hydrogen_generation_diagnostics": diagnostics,
        }
    )
    return replace(
        result,
        generated_hydrogens=generated,
        validation=validation,
    )
