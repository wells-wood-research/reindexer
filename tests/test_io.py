import numpy as np
import pandas as pd

from reindex.io import df2pdb, pdb2df


def test_pdb_round_trip_preserves_atom_identity_and_coordinates(tmp_path):
    source = pd.DataFrame(
        [
            {
                "ATOM": "HETATM",
                "ATOM_ID": 7,
                "ATOM_NAME": "C1",
                "RES_NAME": "LIG",
                "CHAIN_ID": "A",
                "RES_ID": 3,
                "X": 1.234,
                "Y": -2.345,
                "Z": 3.456,
                "OCCUPANCY": 1.0,
                "BETAFACTOR": 0.0,
                "ELEMENT": "C",
            }
        ]
    )
    path = tmp_path / "roundtrip.pdb"

    assert df2pdb(source, path) == path
    parsed = pdb2df(path)

    assert parsed is not None
    assert list(parsed["ATOM_NAME"]) == ["C1"]
    assert list(parsed["ELEMENT"]) == ["C"]
    assert np.allclose(parsed[["X", "Y", "Z"]].to_numpy(), [[1.234, -2.345, 3.456]])
