"""The reindexer reads a molecule's component CIF directly, in the file's atom order.

A component CIF defines a molecule rather than modelling one: its atoms are the
rows of ``_chem_comp_atom``, it has no ``_atom_site`` loop, and an atom is
addressed by its 0-based row position. prometheozyme's graph configs, assembly
components and ORCA inputs all use that same number, so the reindexer's output
must come out in exactly that order -- an off-by-one here would silently repoint
every restraint downstream.

Until this reader existed, prometheozyme converted every molecule CIF to a PDB
before each reindexing run (``sanitise._reference_pdb``), only because the
reindexer could read PDB and XYZ alone. These tests are what allow that
round-trip to be removed.

Two flavours are in use and both are fixtures here, copied unmodified:
``arginine.cif`` is an 8-column library fragment (Cbeta-capped, XTB-optimised
ideal coordinates, no pdbx_ordinal); ``RMN.cif`` is a 21-column wwPDB entry
whose ``model_Cartn_*`` and ``pdbx_model_Cartn_*_ideal`` columns are different
geometries, of which only the ideal set is the definition's.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from reindexer import reindex
from reindexer.io import cif2df, df2pdb, pdb2df


FIXTURES = Path(__file__).parent / "fixture_files"
ARGININE = FIXTURES / "arginine.cif"
RMN = FIXTURES / "RMN.cif"


def _atom_loop(path: Path) -> tuple[list[str], list[list[str]]]:
    """The _chem_comp_atom loop by plain whitespace split: an independent oracle.

    Valid for these two fixtures only, which quote nothing in that loop.
    """
    lines = path.read_text().splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("_chem_comp_atom.")
    )
    tags = []
    index = start
    while lines[index].startswith("_chem_comp_atom."):
        tags.append(lines[index].split(".", 1)[1].strip())
        index += 1
    rows = []
    while not lines[index].startswith(("#", "loop_", "_")):
        rows.append(lines[index].split())
        index += 1
    return tags, rows


def _column(path: Path, name: str) -> list[str]:
    tags, rows = _atom_loop(path)
    return [row[tags.index(name)] for row in rows]


def test_library_fragment_is_read_in_file_order():
    df = cif2df(ARGININE)

    assert list(df["ATOM_NAME"]) == _column(ARGININE, "atom_id")
    assert list(df["ATOM_ID"]) == list(range(1, len(df) + 1))
    expected = np.array(
        [[float(v) for v in xyz] for xyz in zip(
            *(_column(ARGININE, f"pdbx_model_Cartn_{d}_ideal") for d in "xyz")
        )]
    )
    assert np.allclose(df[["X", "Y", "Z"]].to_numpy(), expected)
    assert set(df["ATOM"]) == {"ATOM"}
    assert df.attrs["formal_charge"] == 1
    assert sum(df.attrs["charges"]) == 1


def test_wwpdb_entry_uses_ideal_coordinates_not_model():
    df = cif2df(RMN)

    names = _column(RMN, "atom_id")
    assert list(df["ATOM_NAME"]) == names
    # pdbx_ordinal agrees with row order in this file; it must still be row
    # order that the frame follows.
    assert [int(v) for v in _column(RMN, "pdbx_ordinal")] == list(range(1, len(names) + 1))
    ideal = np.array([[float(v) for v in xyz] for xyz in zip(
        *(_column(RMN, f"pdbx_model_Cartn_{d}_ideal") for d in "xyz")
    )])
    model = np.array([[float(v) for v in xyz] for xyz in zip(
        *(_column(RMN, f"model_Cartn_{d}") for d in "xyz")
    )])
    assert not np.allclose(ideal, model)
    assert np.allclose(df[["X", "Y", "Z"]].to_numpy(), ideal)
    assert set(df["ATOM"]) == {"HETATM"}
    # O12 was corrected to -1 in this file; the charge column must be read by
    # name, not by position.
    assert df.attrs["charges"][names.index("O12")] == -1
    assert df.attrs["formal_charge"] == 0


def _rewrite_atom_loop(path: Path, out: Path, transform) -> Path:
    """Write `path` with its _chem_comp_atom loop replaced by transform(tags, rows)."""
    tags, rows = _atom_loop(path)
    lines = path.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("_chem_comp_atom."))
    end = start + len(tags) + len(rows)
    new_tags, new_rows = transform(list(tags), [list(r) for r in rows])
    replacement = [f"_chem_comp_atom.{t}" for t in new_tags] + [" ".join(r) for r in new_rows]
    out.write_text("\n".join(lines[:start] + replacement + lines[end:]) + "\n")
    return out


def test_columns_are_found_by_header_not_position(tmp_path):
    """Reversing the column order must change nothing."""
    reversed_cif = _rewrite_atom_loop(
        RMN, tmp_path / "RMN.cif",
        lambda tags, rows: (tags[::-1], [row[::-1] for row in rows]),
    )

    pd.testing.assert_frame_equal(cif2df(reversed_cif), cif2df(RMN))


def test_an_ordinal_that_disagrees_with_row_order_is_refused(tmp_path):
    def swap_ordinals(tags, rows):
        column = tags.index("pdbx_ordinal")
        rows[0][column], rows[1][column] = rows[1][column], rows[0][column]
        return tags, rows

    cif = _rewrite_atom_loop(RMN, tmp_path / "RMN.cif", swap_ordinals)

    with pytest.raises(ValueError, match="pdbx_ordinal disagrees with row order"):
        cif2df(cif)


def test_a_model_with_atom_site_is_refused(tmp_path):
    cif = tmp_path / "model.cif"
    cif.write_text(
        ARGININE.read_text()
        + "loop_\n_atom_site.id\n_atom_site.type_symbol\n1 C\n"
    )

    with pytest.raises(ValueError, match="_atom_site"):
        cif2df(cif)


def test_incomplete_ideal_set_falls_back_to_model_and_says_so(tmp_path, caplog):
    def blank_one_ideal(tags, rows):
        rows[3][tags.index("pdbx_model_Cartn_x_ideal")] = "?"
        return tags, rows

    cif = _rewrite_atom_loop(RMN, tmp_path / "RMN.cif", blank_one_ideal)
    with caplog.at_level(logging.WARNING):
        df = cif2df(cif)

    model = np.array([[float(v) for v in xyz] for xyz in zip(
        *(_column(RMN, f"model_Cartn_{d}") for d in "xyz")
    )])
    assert np.allclose(df[["X", "Y", "Z"]].to_numpy(), model)
    assert "C4" in caplog.text and "model_Cartn" in caplog.text


def test_an_atom_unplaced_in_every_set_is_refused(tmp_path):
    # The library flavour has no model set to fall back to.
    def blank(tags, rows):
        rows[0][tags.index("pdbx_model_Cartn_y_ideal")] = "?"
        return tags, rows

    cif = _rewrite_atom_loop(ARGININE, tmp_path / "arginine.cif", blank)

    with pytest.raises(ValueError, match="CB"):
        cif2df(cif)


def test_quoted_atom_names_are_unquoted(tmp_path):
    """Nucleotide-style names such as C1' arrive double-quoted."""
    def quote(tags, rows):
        rows[0][tags.index("atom_id")] = "\"C1'\""
        return tags, rows

    cif = _rewrite_atom_loop(ARGININE, tmp_path / "arginine.cif", quote)

    assert cif2df(cif)["ATOM_NAME"].iloc[0] == "C1'"


def test_reindexing_against_a_cif_keeps_the_cif_atom_order(tmp_path):
    """End to end: a shuffled target comes back in the definition's order.

    The target is the definition itself, rigidly moved and with its rows
    reversed, so every heavy atom's correct position is known.
    """
    reference = cif2df(ARGININE)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = reference.iloc[::-1].reset_index(drop=True).copy()
    target[["X", "Y", "Z"]] = target[["X", "Y", "Z"]].to_numpy() @ rotation.T + 5.0
    target["ATOM_ID"] = range(1, len(target) + 1)
    target_path = df2pdb(target, tmp_path / "target.pdb")
    output_path = tmp_path / "out.pdb"

    reindex(ARGININE, target_path, output_path)

    output = pdb2df(output_path)
    assert list(output["ATOM_NAME"]) == list(reference["ATOM_NAME"])
    heavy = output["ELEMENT"] != "H"
    moved = reference[["X", "Y", "Z"]].to_numpy() @ rotation.T + 5.0
    assert np.allclose(
        output.loc[heavy, ["X", "Y", "Z"]].to_numpy(), moved[heavy.to_numpy()], atol=1e-3
    )
