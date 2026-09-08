import pandas as pd
import pytest

from reindexer.graph import df2graph
from reindexer.matching import GraphMismatchError, MappingAmbiguityError
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


def test_symmetric_mapping_is_not_silently_guessed():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "C", "X": 1.5, "Y": 0.0, "Z": 0.0},
        ]
    )

    with pytest.raises(MappingAmbiguityError) as exc_info:
        reindex_graphs(reference, target)

    assert len(exc_info.value.candidates) == 2


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
