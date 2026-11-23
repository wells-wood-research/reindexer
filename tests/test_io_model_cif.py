"""Reading a model CIF, and writing one, so the cycle need not pass through PDB.

`cif2df` reads a *component definition* -- one block, one component, ideal
coordinates, a row order that is itself the answer. A co-folded prediction is a
different file: an `_atom_site` loop, many chains, real coordinates, and two
parallel numbering schemes. It is read by `model_cif2df` and refused by
`cif2df` on purpose, because one function guarding two sets of assumptions is
how a silent renumbering gets in.

`read_structure` therefore chooses by **contents**, not by suffix: both arrive
as `.cif` and the caller usually does not know which it holds.
"""

from pathlib import Path

import pandas as pd
import pytest

from reindexer.io import (
    PDB_COLUMNS,
    cif_is_model,
    df2cif,
    model_cif2df,
    read_structure,
    write_structure,
)

FIXTURES = Path(__file__).parent / "fixture_files"


def _model_cif(path: Path, body: str, tags=None) -> Path:
    tags = tags or [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
        "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
    ]
    header = "data_test\n#\nloop_\n" + "".join(f"_atom_site.{t}\n" for t in tags)
    path.write_text(header + body + "#\n")
    return path


SIMPLE = (
    "ATOM 1 N N GLY A 1 1.000 2.000 3.000\n"
    "ATOM 2 C CA GLY A 1 1.500 2.500 3.500\n"
    "HETATM 3 O O1 LIG B . 4.000 5.000 6.000\n"
)


# ---------------------------------------------------------------------------
# Telling the two kinds apart
# ---------------------------------------------------------------------------

def test_a_component_definition_is_not_a_model():
    assert cif_is_model(FIXTURES / "arginine.cif") is False


def test_a_file_with_atom_site_is_a_model(tmp_path):
    assert cif_is_model(_model_cif(tmp_path / "m.cif", SIMPLE)) is True


def test_read_structure_sends_each_kind_to_its_own_reader(tmp_path):
    """The dispatch that matters: a prediction and a molecule definition reach
    the same argument, and the caller does not know which it has."""
    component = read_structure(FIXTURES / "arginine.cif")
    model = read_structure(_model_cif(tmp_path / "m.cif", SIMPLE))

    assert list(component.columns) == list(PDB_COLUMNS)
    assert list(model.columns) == list(PDB_COLUMNS)
    assert len(model) == 3


def test_the_component_reader_still_refuses_a_model(tmp_path):
    from reindexer.io import cif2df

    with pytest.raises(ValueError, match="_atom_site"):
        cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))


def test_the_model_reader_refuses_a_component():
    with pytest.raises(ValueError, match="no _atom_site"):
        model_cif2df(FIXTURES / "arginine.cif")


# ---------------------------------------------------------------------------
# What the model reader makes of a model
# ---------------------------------------------------------------------------

def test_atoms_keep_the_file_s_row_order(tmp_path):
    df = model_cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))
    assert list(df["ATOM_NAME"]) == ["N", "CA", "O1"]
    assert list(df["ATOM_ID"]) == [1, 2, 3]


def test_hetatm_is_preserved(tmp_path):
    df = model_cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))
    assert list(df["ATOM"]) == ["ATOM", "ATOM", "HETATM"]


def test_a_ligand_s_absent_seq_id_becomes_zero(tmp_path):
    """`label_seq_id` is `.` for a non-polymer by the dictionary. Zero keeps
    the column integral without inventing an index that could collide."""
    df = model_cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))
    assert list(df["RES_ID"]) == [1, 1, 0]


def test_label_is_preferred_over_auth(tmp_path):
    """`auth_` is whatever a depositor chose and may repeat or skip; `label_`
    is the file's own and cannot collide. The reindexer writes back by row, so
    it needs the one that cannot."""
    tags = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
        "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "auth_atom_id", "auth_comp_id", "auth_asym_id", "auth_seq_id",
    ]
    body = "ATOM 1 N N GLY A 1 1.0 2.0 3.0 ZZ XXX Q 999\n"
    df = model_cif2df(_model_cif(tmp_path / "m.cif", body, tags))

    assert df.loc[0, "ATOM_NAME"] == "N"
    assert df.loc[0, "CHAIN_ID"] == "A"
    assert df.loc[0, "RES_ID"] == 1


def test_auth_is_used_when_label_is_absent_and_says_so(tmp_path, caplog):
    import logging

    tags = [
        "group_PDB", "id", "type_symbol", "auth_atom_id", "auth_comp_id",
        "auth_asym_id", "auth_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
    ]
    with caplog.at_level(logging.WARNING):
        df = model_cif2df(
            _model_cif(tmp_path / "m.cif", "ATOM 1 N N GLY A 7 1.0 2.0 3.0\n", tags)
        )

    assert df.loc[0, "RES_ID"] == 7
    assert any("auth_" in r.getMessage() for r in caplog.records)


def test_more_than_one_model_is_refused_not_guessed(tmp_path):
    """Which model the caller meant is not this reader's guess: reindexing the
    wrong one would renumber against the wrong geometry."""
    tags = [
        "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
        "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
        "pdbx_PDB_model_num",
    ]
    body = (
        "ATOM 1 N N GLY A 1 1.0 2.0 3.0 1\n"
        "ATOM 2 N N GLY A 1 9.0 9.0 9.0 2\n"
    )
    with pytest.raises(ValueError, match="models"):
        model_cif2df(_model_cif(tmp_path / "m.cif", body, tags))


def test_an_empty_atom_site_is_refused_with_the_path_named(tmp_path):
    """biotite says "Failed to deserialize category 'atom_site'" and names
    neither the file nor the reason. This package's contract is that an
    unreadable CIF raises naming why."""
    path = _model_cif(tmp_path / "m.cif", "")

    with pytest.raises(ValueError) as caught:
        model_cif2df(path)

    assert str(path) in str(caught.value)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def test_a_model_round_trips_exactly(tmp_path):
    """Names, order, chains and coordinates all survive. The coordinates are
    the point: three decimals is a PDB column-width limit, and writing it here
    would re-import the loss this phase exists to remove."""
    source = _model_cif(tmp_path / "m.cif", SIMPLE)
    original = model_cif2df(source)

    written = df2cif(original, tmp_path / "out.cif")
    back = model_cif2df(written)

    for column in PDB_COLUMNS:
        assert list(original[column]) == list(back[column]), column


def test_five_decimals_survive(tmp_path):
    body = "ATOM 1 N N GLY A 1 21.23560 10.23518 7.47372\n"
    original = model_cif2df(_model_cif(tmp_path / "m.cif", body))
    back = model_cif2df(df2cif(original, tmp_path / "out.cif"))

    assert back.loc[0, "X"] == pytest.approx(21.23560, abs=0)
    assert back.loc[0, "Z"] == pytest.approx(7.47372, abs=0)


def test_write_structure_follows_the_filename(tmp_path):
    df = model_cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))

    assert write_structure(df, tmp_path / "a.cif").read_text().startswith("data_")
    assert write_structure(df, tmp_path / "a.pdb").read_text().startswith("ATOM")
    with pytest.raises(ValueError, match="Unsupported"):
        write_structure(df, tmp_path / "a.xyz")


def test_an_empty_frame_is_refused_not_written(tmp_path):
    assert df2cif(pd.DataFrame(columns=PDB_COLUMNS), tmp_path / "empty.cif") is None


def test_a_name_with_a_space_is_quoted(tmp_path):
    """An unquoted value with whitespace would silently become two columns."""
    df = model_cif2df(_model_cif(tmp_path / "m.cif", SIMPLE))
    df.loc[0, "RES_NAME"] = "A B"

    back = model_cif2df(df2cif(df, tmp_path / "out.cif"))
    assert back.loc[0, "RES_NAME"] == "A B"


# ---------------------------------------------------------------------------
# Readable by something other than this package
# ---------------------------------------------------------------------------

def test_what_is_written_is_readable_by_biotite(tmp_path):
    """A format only its own author can read is worse than the one it replaces.

    The round-trip tests above use this package's reader on this package's
    writer, which cannot catch a missing column that every *other* reader
    assumes. biotite raised `KeyError: 'pdbx_PDB_ins_code'` on the first
    version of this writer, and prometheozyme reads these files through
    biotite -- so the whole phase would have shipped a CIF the pipeline could
    not load.
    """
    biotite_pdbx = pytest.importorskip("biotite.structure.io.pdbx")

    source = _model_cif(tmp_path / "m.cif", SIMPLE)
    written = df2cif(model_cif2df(source), tmp_path / "out.cif")

    array = biotite_pdbx.get_structure(
        biotite_pdbx.CIFFile.read(str(written)), model=1
    )

    assert array.array_length() == 3
    assert [str(name) for name in array.atom_name] == ["N", "CA", "O1"]
    assert [str(res) for res in array.res_name] == ["GLY", "GLY", "LIG"]
    assert list(array.hetero) == [False, False, True]
