from pathlib import Path

import numpy as np
import pytest

from reindex.graph import pdb2graph
from reindex.hydrogens import generate_missing_hydrogens
from reindex.io import pdb2df
from reindex.pipeline import reindex_graphs
from reindex.writer import (
    PDBWriteError,
    build_reindexed_pdb_dataframe,
    write_reindexed_pdb,
)
FIXTURES = Path(__file__).parent / "fixture_files"


@pytest.mark.parametrize(
    "reference_name,target_name",
    [
        ("fad_reference.pdb", "fad_target_01.pdb"),
        ("fad_reference.pdb", "fad_target_02.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_01.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_02.pdb"),
    ],
)
def test_build_reindexed_dataframe_preserves_reference_slots_and_target_coordinates(
    reference_name, target_name
):
    reference_df = pdb2df(FIXTURES / reference_name)
    target_df = pdb2df(FIXTURES / target_name)
    reference_graph = pdb2graph(FIXTURES / reference_name)
    target_graph = pdb2graph(FIXTURES / target_name)
    result = generate_missing_hydrogens(
        reference_graph,
        target_graph,
        reindex_graphs(reference_graph, target_graph),
    )

    output_df = build_reindexed_pdb_dataframe(reference_df, target_df, result)

    assert list(output_df["ATOM_NAME"]) == list(reference_df["ATOM_NAME"])
    assert list(output_df["ATOM_ID"]) == list(reference_df["ATOM_ID"])
    assert list(output_df["ELEMENT"]) == list(reference_df["ELEMENT"])
    for reference_index, reference_row in reference_df.iterrows():
        if str(reference_row["ELEMENT"]).strip().upper() == "H":
            continue
        target_index = result.reference_to_target[reference_index]
        expected = target_df.iloc[target_index][["X", "Y", "Z"]].to_numpy(dtype=float)
        actual = output_df.iloc[reference_index][["X", "Y", "Z"]].to_numpy(dtype=float)
        assert np.array_equal(actual, expected)

    for reference_index, placement in result.generated_hydrogens.items():
        actual = output_df.iloc[reference_index][["X", "Y", "Z"]].to_numpy(dtype=float)
        assert np.array_equal(actual, placement.coords)


@pytest.mark.parametrize(
    "reference_name,target_name",
    [
        ("fad_reference.pdb", "fad_target_02.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_02.pdb"),
    ],
)
def test_write_reindexed_pdb_round_trips_and_validates_connectivity(
    tmp_path, reference_name, target_name
):
    reference_df = pdb2df(FIXTURES / reference_name)
    target_df = pdb2df(FIXTURES / target_name)
    reference_graph = pdb2graph(FIXTURES / reference_name)
    target_graph = pdb2graph(FIXTURES / target_name)
    result = generate_missing_hydrogens(
        reference_graph,
        target_graph,
        reindex_graphs(reference_graph, target_graph),
    )
    output_path = tmp_path / f"{target_name}.reindexed.pdb"

    returned_path = write_reindexed_pdb(
        reference_df,
        target_df,
        result,
        output_path,
        reference_graph=reference_graph,
        target_graph=target_graph,
    )

    assert returned_path == output_path
    parsed = pdb2df(output_path)
    assert len(parsed) == len(reference_df)
    assert list(parsed["ATOM_NAME"]) == list(reference_df["ATOM_NAME"])
    assert list(parsed["ELEMENT"]) == list(reference_df["ELEMENT"])
    for reference_index, reference_row in reference_df.iterrows():
        if str(reference_row["ELEMENT"]).strip().upper() == "H":
            continue
        target_index = result.reference_to_target[reference_index]
        expected = target_df.iloc[target_index][["X", "Y", "Z"]].to_numpy(dtype=float)
        actual = parsed.iloc[reference_index][["X", "Y", "Z"]].to_numpy(dtype=float)
        assert np.allclose(actual, expected, atol=5.1e-4, rtol=0.0)


def test_writer_rejects_missing_generated_slots(tmp_path):
    reference_graph = pdb2graph(FIXTURES / "palmitic_acid_reference.pdb")
    target_graph = pdb2graph(FIXTURES / "palmitic_acid_target_02.pdb")
    result = reindex_graphs(reference_graph, target_graph)
    reference_df = pdb2df(FIXTURES / "palmitic_acid_reference.pdb")
    target_df = pdb2df(FIXTURES / "palmitic_acid_target_02.pdb")

    with pytest.raises(PDBWriteError, match="no target or generated coordinate"):
        write_reindexed_pdb(
            reference_df,
            target_df,
            result,
            tmp_path / "incomplete.pdb",
            reference_graph=reference_graph,
            target_graph=target_graph,
        )
