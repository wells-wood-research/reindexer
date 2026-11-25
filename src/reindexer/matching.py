"""NetworkX heavy-atom graph matching and ambiguity resolution."""

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx
import numpy as np

LOG = logging.getLogger(__name__)

#: How far a relabelled copy of the reference may sit from the reference itself and
#: still count as the same structure. A true symmetry operation superimposes exactly;
#: this is slack for coordinate round-tripping through a CIF, not for near-symmetry.
SYMMETRY_RMSD_TOLERANCE = 0.05


class GraphMismatchError(ValueError):
    """Raised when reference and target heavy graphs cannot be isomorphic."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"Heavy-atom graphs are not isomorphic: {diagnostics}")


class MappingAmbiguityError(ValueError):
    """Raised when graph-equivalent mappings remain tied after all tie-breakers."""

    def __init__(self, candidates: list[dict[int, int]], scores: dict[str, Any]):
        self.candidates = candidates
        self.scores = scores
        super().__init__(
            f"Heavy-atom mapping remains ambiguous after all tie-breakers; "
            f"{len(candidates)} candidates remain"
        )


@dataclass
class MappingOutcome:
    mapping: dict[int, int]
    candidates: list[dict[int, int]]
    scores: dict[str, Any]


def _node_signature(graph: nx.Graph, node: int) -> tuple:
    data = graph.nodes[node]
    neighbors = list(graph.neighbors(node))
    neighbor_pairs = tuple(
        sorted(
            (
                graph.nodes[neighbor]["element"],
                graph.degree[neighbor],
            )
            for neighbor in neighbors
        )
    )
    neighbor_elements = tuple(sorted(graph.nodes[n]["element"] for n in neighbors))
    return (
        data["element"],
        graph.degree[node],
        neighbor_elements,
        neighbor_pairs,
    )


def add_connectivity_signatures(graph: nx.Graph) -> nx.Graph:
    """Attach invariant structural signatures to a graph copy."""
    result = graph.copy()
    for node in result:
        result.nodes[node]["connectivity_signature"] = _node_signature(result, node)
    return result


def heavy_atom_graph(graph: nx.Graph) -> nx.Graph:
    """Return a heavy-only view while preserving original node identifiers."""
    heavy_nodes = [node for node, data in graph.nodes(data=True) if data["element"] != "H"]
    return add_connectivity_signatures(graph.subgraph(heavy_nodes).copy())


def _graph_diagnostics(reference: nx.Graph, target: nx.Graph) -> dict[str, Any]:
    def stats(graph: nx.Graph) -> dict[str, Any]:
        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "element_histogram": dict(
                sorted(Counter(nx.get_node_attributes(graph, "element").values()).items())
            ),
            "degree_distribution": dict(sorted(Counter(dict(graph.degree()).values()).items())),
            "component_sizes": sorted((len(component) for component in nx.connected_components(graph)), reverse=True),
        }

    return {
        "reference": stats(reference),
        "target": stats(target),
        "element_histogram_match": Counter(
            nx.get_node_attributes(reference, "element").values()
        )
        == Counter(nx.get_node_attributes(target, "element").values()),
        "degree_distribution_match": Counter(dict(reference.degree()).values())
        == Counter(dict(target.degree()).values()),
    }


def _close(value: float, best: float, atol: float = 1e-9) -> bool:
    return bool(np.isclose(value, best, rtol=1e-7, atol=atol))


def _retain_best(
    candidates: list[dict[int, int]],
    score: Callable[[dict[int, int]], float],
) -> tuple[list[dict[int, int]], float]:
    values = [float(score(candidate)) for candidate in candidates]
    best = min(values)
    return [candidate for candidate, value in zip(candidates, values) if _close(value, best)], best


def _bond_distance_score(target: nx.Graph, reference: nx.Graph, mapping: dict[int, int]) -> float:
    score = 0.0
    for target_a, target_b, edge in target.edges(data=True):
        reference_a = mapping[target_a]
        reference_b = mapping[target_b]
        reference_edge = reference.get_edge_data(reference_a, reference_b)
        score += abs(float(edge["distance"]) - float(reference_edge["distance"]))
    return score


def _angle_score(target: nx.Graph, reference: nx.Graph, mapping: dict[int, int]) -> float:
    score = 0.0
    for target_center in target.nodes:
        target_neighbors = list(target.neighbors(target_center))
        reference_center = mapping[target_center]
        reference_neighbors = [mapping[n] for n in target_neighbors]
        target_center_coords = target.nodes[target_center]["coords"]
        reference_center_coords = reference.nodes[reference_center]["coords"]
        for i, target_a in enumerate(target_neighbors):
            target_vector_a = target.nodes[target_a]["coords"] - target_center_coords
            reference_vector_a = reference.nodes[reference_neighbors[i]]["coords"] - reference_center_coords
            norm_target_a = np.linalg.norm(target_vector_a)
            norm_reference_a = np.linalg.norm(reference_vector_a)
            if norm_target_a == 0 or norm_reference_a == 0:
                continue
            for j in range(i + 1, len(target_neighbors)):
                target_b = target_neighbors[j]
                target_vector_b = target.nodes[target_b]["coords"] - target_center_coords
                reference_vector_b = reference.nodes[reference_neighbors[j]]["coords"] - reference_center_coords
                norm_target_b = np.linalg.norm(target_vector_b)
                norm_reference_b = np.linalg.norm(reference_vector_b)
                if norm_target_b == 0 or norm_reference_b == 0:
                    continue
                target_cos = np.dot(target_vector_a, target_vector_b) / (norm_target_a * norm_target_b)
                reference_cos = np.dot(reference_vector_a, reference_vector_b) / (
                    norm_reference_a * norm_reference_b
                )
                score += abs(float(np.clip(target_cos, -1, 1)) - float(np.clip(reference_cos, -1, 1)))
    return score


def _kabsch_rmsd(target: nx.Graph, reference: nx.Graph, mapping: dict[int, int]) -> float:
    target_nodes = sorted(mapping)
    target_coords = np.asarray([target.nodes[node]["coords"] for node in target_nodes], dtype=float)
    reference_coords = np.asarray(
        [reference.nodes[mapping[node]]["coords"] for node in target_nodes], dtype=float
    )
    target_centered = target_coords - target_coords.mean(axis=0)
    reference_centered = reference_coords - reference_coords.mean(axis=0)
    covariance = target_centered.T @ reference_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt)) or 1.0
    rotation = u @ correction @ vt
    aligned = target_centered @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - reference_centered) ** 2, axis=1))))


def _permutation_is_reference_symmetry(reference: nx.Graph, permutation: dict[int, int]) -> float:
    """RMSD between the reference and the same reference with its nodes relabelled.

    A permutation that superimposes on the original is a symmetry operation of the
    molecule: the atoms it exchanges are the same atom chemically and geometrically,
    so a mapping that differs from another only by it names the same structure.
    Formate is the case in point -- its two carboxylate oxygens carry no hydrogen and
    sit at equal C-O distances, so no bond-length, angle or RMSD tie-breaker can tell
    the two mappings apart, because there is nothing to tell apart.

    Returned as an RMSD rather than a boolean so the caller can record how exact the
    symmetry was.
    """
    nodes = sorted(permutation)
    original = np.asarray([reference.nodes[node]["coords"] for node in nodes], dtype=float)
    relabelled = np.asarray(
        [reference.nodes[permutation[node]]["coords"] for node in nodes], dtype=float
    )
    original_centered = original - original.mean(axis=0)
    relabelled_centered = relabelled - relabelled.mean(axis=0)
    covariance = relabelled_centered.T @ original_centered
    u, _, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt)) or 1.0
    aligned = relabelled_centered @ (u @ correction @ vt)
    return float(np.sqrt(np.mean(np.sum((aligned - original_centered) ** 2, axis=1))))


def match_heavy_atoms(reference: nx.Graph, target: nx.Graph, *, allow_subgraph: bool = False) -> MappingOutcome:
    """Match reference to target, optionally allowing target-only atoms."""
    reference = add_connectivity_signatures(reference)
    target = add_connectivity_signatures(target)
    diagnostics = _graph_diagnostics(reference, target)
    if diagnostics["target"]["node_count"] < diagnostics["reference"]["node_count"]:
        raise GraphMismatchError(diagnostics)
    if not allow_subgraph and (
        diagnostics["reference"]["node_count"] != diagnostics["target"]["node_count"]
        or not diagnostics["element_histogram_match"]
        or not diagnostics["degree_distribution_match"]
    ):
        raise GraphMismatchError(diagnostics)

    if allow_subgraph and reference.number_of_nodes() < target.number_of_nodes():
        node_match = nx.algorithms.isomorphism.categorical_node_match("element", None)
        matcher = nx.algorithms.isomorphism.GraphMatcher(target, reference, node_match=node_match)
        candidates = [dict(mapping) for mapping in matcher.subgraph_isomorphisms_iter()]
    else:
        node_match = nx.algorithms.isomorphism.categorical_node_match("connectivity_signature", None)
        matcher = nx.algorithms.isomorphism.GraphMatcher(target, reference, node_match=node_match)
        candidates = [dict(mapping) for mapping in matcher.isomorphisms_iter()]
    candidates.sort(key=lambda mapping: tuple(mapping[node] for node in sorted(mapping)))
    diagnostics["isomorphism_candidate_count"] = len(candidates)
    if not candidates:
        raise GraphMismatchError(diagnostics)

    scores: dict[str, Any] = {
        "candidate_count": len(candidates),
        "connectivity_signature_survivors": len(candidates),
    }
    candidates, best_bond = _retain_best(
        candidates, lambda mapping: _bond_distance_score(target.subgraph(mapping), reference, mapping)
    )
    scores["bond_distance_best"] = best_bond
    scores["bond_distance_survivors"] = len(candidates)

    candidates, best_angle = _retain_best(
        candidates, lambda mapping: _angle_score(target.subgraph(mapping), reference, mapping)
    )
    scores["local_geometry_best"] = best_angle
    scores["local_geometry_survivors"] = len(candidates)

    candidates, best_rmsd = _retain_best(
        candidates, lambda mapping: _kabsch_rmsd(target.subgraph(mapping), reference, mapping)
    )
    scores["global_rmsd_best"] = best_rmsd
    scores["global_rmsd_survivors"] = len(candidates)
    if len(candidates) != 1:
        # Every tie-breaker above is geometric, so candidates related by a symmetry
        # operation of the reference survive all of them and always will: they
        # describe the same structure under two equally correct namings. Picking the
        # first is then a choice of label, not a guess about the structure, and it is
        # made deterministically because `candidates` is sorted. A tie that is NOT a
        # symmetry is still refused -- equal scores do not imply the relabelling
        # superimposes, so this check is a real one.
        chosen = candidates[0]
        deviations = [
            _permutation_is_reference_symmetry(
                reference, {chosen[node]: candidate[node] for node in chosen}
            )
            for candidate in candidates[1:]
        ]
        worst = max(deviations) if deviations else 0.0
        scores["symmetry_equivalent_candidates"] = len(candidates)
        scores["symmetry_rmsd_worst"] = worst
        if worst > SYMMETRY_RMSD_TOLERANCE:
            raise MappingAmbiguityError(candidates, scores)
        LOG.warning(
            "%d heavy-atom mappings are tied and differ only by a symmetry of the "
            "reference (worst relabelling RMSD %.4f A, tolerance %.2f A); taking the "
            "first by sorted order. The atoms they exchange are the same atom "
            "chemically, so either naming is correct.",
            len(candidates), worst, SYMMETRY_RMSD_TOLERANCE,
        )
        return MappingOutcome(chosen, candidates, scores)
    return MappingOutcome(candidates[0], candidates, scores)
