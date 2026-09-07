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
