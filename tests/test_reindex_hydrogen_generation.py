from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reindex.graph import df2graph, pdb2graph
from reindex.hydrogens import HYDROGEN_BOND_LENGTHS, generate_missing_hydrogens
from reindex.pipeline import reindex_graphs


def _tetrahedral_methyl_rows():
    rows = [
        {"ELEMENT": "C", "X": 0.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C1"},
        {"ELEMENT": "C", "X": 1.54, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "C2"},
        {"ELEMENT": "O", "X": 3.0, "Y": 0.0, "Z": 0.0, "ATOM_NAME": "O3"},
    ]
    axis = np.array([-1.0, 0.0, 0.0])
    first = np.array([0.0, 1.0, 0.0])
    second = np.array([0.0, 0.0, 1.0])
    for index, angle in enumerate((0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0), start=1):
        direction = axis / 3.0 + np.sqrt(8.0 / 9.0) * (
            np.cos(angle) * first + np.sin(angle) * second
        )
        rows.append(
            {
                "ELEMENT": "H",
                "X": float(1.09 * direction[0]),
                "Y": float(1.09 * direction[1]),
                "Z": float(1.09 * direction[2]),
                "ATOM_NAME": f"H{index}",
            }
        )
    return rows


def test_tetrahedral_generation_is_deterministic_and_reference_indexed():
    reference = df2graph(pd.DataFrame(_tetrahedral_methyl_rows()))
    target = df2graph(pd.DataFrame(_tetrahedral_methyl_rows()[:3]))

    mapped = reindex_graphs(reference, target)
    placed = generate_missing_hydrogens(reference, target, mapped)

    assert sorted(placed.generated_hydrogens) == [3, 4, 5]
    coordinates = [placement.coords for placement in placed.generated_hydrogens.values()]
    assert all(
        np.linalg.norm(coords - target.nodes[0]["coords"]) == pytest.approx(1.09)
        for coords in coordinates
    )
    directions = [coords / np.linalg.norm(coords) for coords in coordinates]
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            assert np.dot(directions[first_index], directions[second_index]) == pytest.approx(-1.0 / 3.0)
    assert placed.validation["generated_hydrogen_slots_complete"] is True


def test_only_missing_hydrogens_are_generated_and_existing_coordinates_are_unchanged():
    rows = _tetrahedral_methyl_rows()
    reference = df2graph(pd.DataFrame(rows))
    target_rows = [rows[2], rows[1], rows[0], rows[3]]
    target = df2graph(pd.DataFrame(target_rows))
    existing_coords = target.nodes[3]["coords"].copy()

    mapped = reindex_graphs(reference, target)
    placed = generate_missing_hydrogens(reference, target, mapped)

    assert len(placed.generated_hydrogens) == 2
    assert placed.missing_hydrogens[2] == 2
    assert np.array_equal(target.nodes[3]["coords"], existing_coords)
    assert placed.target_hydrogen_to_reference == {3: 3}
    assert set(placed.generated_hydrogens) == {4, 5}


@pytest.mark.parametrize(
    "reference_name,target_name",
    [
        ("fad_reference.pdb", "fad_target_01.pdb"),
        ("fad_reference.pdb", "fad_target_02.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_01.pdb"),
        ("palmitic_acid_reference.pdb", "palmitic_acid_target_02.pdb"),
    ],
)
def test_target_generates_every_reference_hydrogen_without_reference_coordinates(
    reference_name, target_name
):
    root = Path(__file__).parent / "fixture_files"
    reference = pdb2graph(root / reference_name)
    target = pdb2graph(root / target_name)
    target_heavy_coordinates = {
        node: target.nodes[node]["coords"].copy()
        for node, data in target.nodes(data=True)
        if data["element"] != "H"
    }
    reference_hydrogen_coordinates = [
        reference.nodes[node]["coords"]
        for node, data in reference.nodes(data=True)
        if data["element"] == "H"
    ]
    reference_hydrogen_indices = {
        node for node, data in reference.nodes(data=True) if data["element"] == "H"
    }

    placed = generate_missing_hydrogens(
        reference,
        target,
        reindex_graphs(reference, target),
    )

    assert len(placed.generated_hydrogens) == 31
    assert set(placed.generated_hydrogens) == reference_hydrogen_indices
    assert all(
        not any(
            np.array_equal(placement.coords, reference_coords)
            for reference_coords in reference_hydrogen_coordinates
        )
        for placement in placed.generated_hydrogens.values()
    )
    for placement in placed.generated_hydrogens.values():
        element = target.nodes[placement.target_parent_index]["element"]
        distance = np.linalg.norm(
            placement.coords - target.nodes[placement.target_parent_index]["coords"]
        )
        assert distance == pytest.approx(HYDROGEN_BOND_LENGTHS.get(element, 1.00))
    for node, coordinates in target_heavy_coordinates.items():
        assert np.array_equal(target.nodes[node]["coords"], coordinates)
