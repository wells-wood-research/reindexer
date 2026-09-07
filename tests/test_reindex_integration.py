from pathlib import Path

import pytest

from reindexer.graph import pdb2graph
from reindexer.pipeline import reindex_graphs


FIXTURES = Path(__file__).parent / "fixture_files"


@pytest.mark.parametrize(
    "reference,target",
    [
        ("fad_reference.pdb", "fad_target_01.pdb"),
        ("fad_reference.pdb", "fad_target_02.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_01.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_02.pdb"),
    ],
)
def test_reference_and_target_are_fully_reindexed(reference, target):
    reference_graph = pdb2graph(FIXTURES / reference)
    target_graph = pdb2graph(FIXTURES / target)

    result = reindex_graphs(reference_graph, target_graph)

    assert result.validation["heavy_isomorphic"] is True
    assert result.validation["one_to_one_heavy_mapping"] is True
    assert result.validation["heavy_atom_count_reference"] == result.validation[
        "heavy_atom_count_target"
    ]
    assert len(result.target_to_reference) == result.validation["heavy_atom_count_reference"]
    assert result.validation["reference_hydrogen_count"] == sum(
        result.expected_hydrogens.values()
    )
    assert sum(result.missing_hydrogens.values()) == result.validation[
        "reference_hydrogen_count"
    ]
    assert result.mapping_scores["global_rmsd_survivors"] == 1
