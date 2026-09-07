import pandas as pd

from reindexer.graph import df2graph
from reindexer.pipeline import reindex_graphs


def _graph(rows):
    return df2graph(pd.DataFrame(rows))


def test_existing_hydrogen_is_assigned_to_reference_slot():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C1"},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H1"},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H1"},
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C1"},
        ]
    )

    result = reindex_graphs(reference, target)

    assert result.missing_hydrogens == {1: 0}
    assert result.existing_hydrogens == {1: [0]}
    assert result.reference_hydrogen_to_target == {1: 0}
    assert result.target_hydrogen_to_reference == {0: 1}
    assert result.validation["hydrogen_slots_complete"] is True


def test_extra_target_hydrogen_is_omitted():
    reference = _graph(
        [{"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0}]
    )
    target = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1},
        ]
    )

    result = reindex_graphs(reference, target)

    assert result.missing_hydrogens == {0: 0}
    assert result.existing_hydrogens == {0: []}
    assert result.reference_hydrogen_to_target == {}
    assert result.validation["omitted_target_hydrogens"] == [1]


def test_surplus_target_hydrogen_is_removed_from_reference_slots():
    reference = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H1"},
        ]
    )
    target = _graph(
        [
            {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1, "ATOM_NAME": "H1"},
            {"ELEMENT": "H", "X": 0.0, "Y": 1.1, "Z": 0.0, "ATOM_NAME": "H2"},
        ]
    )

    result = reindex_graphs(reference, target)

    assert result.missing_hydrogens == {0: 0}
    assert result.existing_hydrogens == {0: [1]}
    assert result.reference_hydrogen_to_target == {1: 1}
    assert result.target_hydrogen_to_reference == {1: 1}
    assert result.validation["omitted_target_hydrogens"] == [2]
