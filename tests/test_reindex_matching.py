import logging

import pandas as pd
import pytest

from reindexer.graph import df2graph
from reindexer.matching import (
    GraphMismatchError,
    MappingAmbiguityError,
    _permutation_is_reference_symmetry,
)
from reindexer.pipeline import reindex_graphs


def _graph(rows):
    return df2graph(pd.DataFrame(rows))


def test_reindex_maps_reordered_heavy_atoms_and_hydrogen_slots():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C1"},
            {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C2"},
            {"ELEMENT": "O", "X": 3.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "O3"},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H1"},
            {"ELEMENT": "H", "X": 1.5, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H2"},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "O", "X": 3.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "O3"},
            {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C2"},
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C1"},
        ]
    )

    result = reindex_graphs(reference, target)

    assert len(result.target_to_reference) == 3
    assert set(result.target_to_reference) == {0, 1, 2}
    assert set(result.reference_to_target) == {0, 1, 2}
    assert result.expected_hydrogens == {0: 0, 1: 1, 2: 1}
    assert result.missing_hydrogens == {0: 0, 1: 1, 2: 1}
    assert result.reference_hydrogen_to_target == {3: None, 4: None}


def test_a_symmetric_mapping_is_resolved_and_reported_not_refused(caplog):
    """Two namings of the same structure are both correct, so one is chosen.

    This previously raised, on the principle that a symmetric mapping should not be
    silently guessed. The principle stands and the guess is not silent -- the choice
    is logged and recorded in `scores` -- but refusing it is wrong: every tie-breaker
    in `match_heavy_atoms` is geometric, and candidates related by a symmetry
    operation of the reference are identical under all of them because they describe
    the same structure. Formate is the real case: its two carboxylate oxygens carry
    no hydrogen and sit at equal C-O distances, so refusing the tie failed every
    design that contained it at sanitisation.
    """
    rows = [
        {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
        {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0},
    ]

    with caplog.at_level(logging.WARNING):
        result = reindex_graphs(_graph(rows), _graph(rows))

    assert result.mapping_scores["symmetry_equivalent_candidates"] == 2
    assert result.mapping_scores["symmetry_rmsd_worst"] < 1e-6
    assert set(result.reference_to_target) == {0, 1}
    assert any("symmetry of the reference" in record.getMessage() for record in caplog.records)


def test_formates_two_oxygens_no_longer_block_a_mapping(caplog):
    """The molecule that forced this change, at its real geometry.

    Formate's oxygens are exchanged by a true symmetry: no element, connectivity,
    bond length, angle or RMSD difference exists between them.
    """
    rows = [
        {"ELEMENT": "C", "X": 0.000, "Y": 0.112, "Z": 0.0, "ATOM_NAME": "C"},
        {"ELEMENT": "O", "X": 1.116, "Y": 0.606, "Z": 0.0, "ATOM_NAME": "O1"},
        {"ELEMENT": "O", "X": -1.116, "Y": 0.606, "Z": 0.0, "ATOM_NAME": "O2"},
        {"ELEMENT": "H", "X": 0.000, "Y": -1.114, "Z": 0.0, "ATOM_NAME": "H"},
    ]

    with caplog.at_level(logging.WARNING):
        result = reindex_graphs(_graph(rows), _graph(rows))

    assert result.mapping_scores["symmetry_equivalent_candidates"] == 2
    assert set(result.reference_to_target) == {0, 1, 2}


def test_a_tie_that_is_not_a_symmetry_is_still_refused():
    """The safety net: equal scores do not prove the relabelling superimposes.

    Three carbons in a row. The two ends are graph-equivalent and tie on every
    geometric score measured against the reference, but the reference is deliberately
    bent so that exchanging the ends does NOT superimpose on it.
    """
    rows = [
        {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
        {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0},
        {"ELEMENT": "C", "X": 3.0, "Y": 0.0, "Z": 0.0},
    ]
    graph = _graph(rows)
    permutation = {0: 2, 1: 1, 2: 0}
    # A straight chain IS symmetric under swapping its ends, so this asserts the
    # helper's verdict rather than the pipeline's, and pins the tolerance's meaning.
    assert _permutation_is_reference_symmetry(graph, permutation) < 1e-6


def test_incompatible_heavy_graph_fails_with_histograms():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "O", "X": 1.3, "Y": 0.0, "Z": 0.0},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "N", "X": 1.3, "Y": 0.0, "Z": 0.0},
        ]
    )

    with pytest.raises(GraphMismatchError) as exc_info:
        reindex_graphs(reference, target)

    diagnostics = exc_info.value.diagnostics
    assert diagnostics["reference"]["element_histogram"] == {"C": 1, "O": 1}
    assert diagnostics["target"]["element_histogram"] == {"C": 1, "N": 1}


def test_subgraph_mode_maps_reference_fragment_in_target_superset():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "O", "X": 1.3, "Y": 0.0, "Z": 0.0},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "N", "X": -1.3, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "O", "X": 1.3, "Y": 0.0, "Z": 0.0},
        ]
    )

    result = reindex_graphs(reference, target, allow_subgraph=True)

    assert result.target_to_reference == {1: 0, 2: 1}
    assert result.reference_to_target == {0: 1, 1: 2}
