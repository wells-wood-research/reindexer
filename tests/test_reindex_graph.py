import pandas as pd
import pytest

from reindexer.graph import df2graph


def test_df2graph_builds_connectivity_and_retains_metadata():
    df = pd.DataFrame(
        [
            {
                "ELEMENT": "C",
                "X": 0.0,
                "Y": 0.0,
                "Z": 0.0,
                "ATOM_ID": 42,
                "ATOM_NAME": "C1",
                "RES_NAME": "LIG",
                "RES_ID": 7,
                "CHAIN_ID": "B",
            },
            {"ELEMENT": "O", "X": 1.3, "Y": 0.0, "Z": 0.0},
            {"ELEMENT": "H", "X": 0.0, "Y": 0.0, "Z": 1.1},
            {"ELEMENT": "C", "X": 4.0, "Y": 0.0, "Z": 0.0},
        ],
        index=[10, 20, 30, 40],
    )

    graph = df2graph(df)

    assert set(graph.nodes) == {0, 1, 2, 3}
    assert graph.nodes[0]["original_index"] == 0
    assert graph.nodes[0]["source_row"] == 10
    assert graph.nodes[0]["pdb_serial"] == 42
    assert graph.nodes[0]["atom_name"] == "C1"
    assert set(graph.edges) == {(0, 1), (0, 2)}
    assert graph[0][1]["distance"] == pytest.approx(1.3)
    assert "bond_order" not in graph[0][1]


def test_df2graph_rejects_missing_columns_and_nonfinite_coordinates():
    with pytest.raises(ValueError, match="missing required columns"):
        df2graph(pd.DataFrame({"ELEMENT": ["C"]}))

    with pytest.raises(ValueError, match="invalid coordinates"):
        df2graph(
            pd.DataFrame(
                {"ELEMENT": ["C"], "X": [float("nan")], "Y": [0.0], "Z": [0.0]}
            )
        )


def test_a_pyrophosphate_bridge_is_not_perceived_as_a_phosphorus_phosphorus_bond():
    """The two phosphorus atoms of a pyrophosphate must not be bonded to each other.

    In FAD the two phosphorus atoms sit 2.584 A apart, bridged through O3P
    rather than bonded directly.  With Cordero's single-bond covalent radius
    for phosphorus, 1.07 A, the cutoff is 1.07 + 1.07 + 0.45 = 2.59 A and that
    non-bonded contact is perceived as an edge.  The spurious edge changes the
    connectivity graph, so no isomorphism is found against the reference and
    the ligand fails to sanitise -- which is what blocked every FAD-bearing
    design until the radius was narrowed to 0.90 A.

    The coordinates are the real ones, taken from a grafted FAD adduct, so the
    test fails against the old radius and would fail again if anyone restored
    it.  It also pins the bonds that must survive the narrower radius: all six
    phosphorus-oxygen bonds, the longest of which is 1.611 A, and neither of
    the two 1-3 contacts from a phosphorus to a carbon across its ester oxygen.
    """
    df = pd.DataFrame(
        [
            {"ELEMENT": "P", "ATOM_NAME": "PA", "X": -3.3215506, "Y": -5.357626, "Z": 11.800551},
            {"ELEMENT": "O", "ATOM_NAME": "O1A", "X": -4.7990932, "Y": -5.4088264, "Z": 11.859692},
            {"ELEMENT": "O", "ATOM_NAME": "O2A", "X": -2.7931368, "Y": -4.0998726, "Z": 12.657215},
            {"ELEMENT": "O", "ATOM_NAME": "O5B", "X": -2.7118773, "Y": -6.7161016, "Z": 12.411797},
            {"ELEMENT": "C", "ATOM_NAME": "C5B", "X": -3.1639595, "Y": -6.799581, "Z": 13.764623},
            {"ELEMENT": "C", "ATOM_NAME": "C5'", "X": -3.6559818, "Y": -2.3751333, "Z": 7.7680573},
            {"ELEMENT": "O", "ATOM_NAME": "O5'", "X": -3.059534, "Y": -3.6015458, "Z": 8.192081},
            {"ELEMENT": "P", "ATOM_NAME": "P", "X": -3.492565, "Y": -3.825095, "Z": 9.726564},
            {"ELEMENT": "O", "ATOM_NAME": "O1P", "X": -3.0048144, "Y": -2.6932364, "Z": 10.5455065},
            {"ELEMENT": "O", "ATOM_NAME": "O2P", "X": -5.0961576, "Y": -3.9069126, "Z": 9.824677},
            {"ELEMENT": "O", "ATOM_NAME": "O3P", "X": -2.850961, "Y": -5.1969543, "Z": 10.271152},
        ]
    )

    graph = df2graph(df)
    name = {node: graph.nodes[node]["atom_name"] for node in graph.nodes}
    bonded = {frozenset((name[a], name[b])) for a, b in graph.edges}

    assert frozenset(("PA", "P")) not in bonded
    assert frozenset(("PA", "C5B")) not in bonded
    assert frozenset(("P", "C5'")) not in bonded
    for phosphorus, oxygen in [
        ("PA", "O1A"), ("PA", "O2A"), ("PA", "O5B"), ("PA", "O3P"),
        ("P", "O1P"), ("P", "O2P"), ("P", "O3P"), ("P", "O5'"),
    ]:
        assert frozenset((phosphorus, oxygen)) in bonded, f"{phosphorus}-{oxygen} lost"
