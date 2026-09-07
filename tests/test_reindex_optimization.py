from pathlib import Path

import pytest

from reindex.io import pdb2df
from reindex.labelling import label_xyz_with_reference
from reindex.orca import write_hydrogen_opt
from reindex.pipeline import reindex


FIXTURES = Path(__file__).parent / "fixture_files"


def _write_xyz_from_pdb(pdb_path, xyz_path, move_hydrogens=False):
    df = pdb2df(pdb_path)
    lines = [f"{len(df)}\n", "mock ORCA output\n"]
    for _, row in df.iterrows():
        z = float(row["Z"])
        if move_hydrogens and str(row["ELEMENT"]).strip().upper() == "H":
            z += 0.123
        lines.append(
            f"{row['ELEMENT']} {float(row['X']):.6f} "
            f"{float(row['Y']):.6f} {z:.6f}\n"
        )
    xyz_path.write_text("".join(lines))


def test_write_hydrogen_opt_writes_exact_input(tmp_path):
    pdb_path = tmp_path / "input.pdb"
    input_path = tmp_path / "opt.inp"
    pdb_path.write_text("PDB\n")

    returned = write_hydrogen_opt(input_path, pdb_path, -1, 2)

    assert returned == input_path
    assert input_path.read_text() == (
        "! XTB2 OptH\n"
        f"*pdbfile -1 2 {pdb_path.resolve()}\n"
    )


def test_label_xyz_with_reference_rejects_element_order_mismatch(tmp_path):
    reference_path = FIXTURES / "palmitic_acid_reference.pdb"
    reference_df = pdb2df(reference_path)
    xyz_path = tmp_path / "mismatch.xyz"
    output_path = tmp_path / "output.pdb"
    elements = list(reference_df["ELEMENT"])
    elements[0], elements[2] = elements[2], elements[0]
    lines = [f"{len(reference_df)}\n", "mismatch\n"]
    for element, (_, row) in zip(elements, reference_df.iterrows()):
        lines.append(f"{element} {row['X']} {row['Y']} {row['Z']}\n")
    xyz_path.write_text("".join(lines))

    with pytest.raises(ValueError, match="element order mismatch"):
        label_xyz_with_reference(reference_path, xyz_path, output_path)


def test_reindex_optimise_branch_uses_mock_orca_and_relabels_output(tmp_path, monkeypatch):
    reference_path = FIXTURES / "palmitic_acid_reference.pdb"
    target_path = FIXTURES / "palmitic_acid_target_02.pdb"
    output_path = tmp_path / "palmitic_acid_reindexed.pdb"

    def mock_run_orca(input_path, orcadir, timeout, logger=None):
        assert input_path.name == "opt.inp"
        assert input_path.read_text().startswith("! XTB2 OptH\n")
        input_path.with_suffix(".out").write_text("mock ORCA stdout\n")
        input_path.with_suffix(".err").write_text("")
        _write_xyz_from_pdb(
            input_path.parent / "reindexed.pdb",
            input_path.parent / "opt.xyz",
            move_hydrogens=True,
        )
        return 0

    monkeypatch.setattr("reindex.optimization.orca.run_orca", mock_run_orca)

    result = reindex(
        reference_path,
        target_path,
        output_path,
        optimise=True,
        orcadir=tmp_path,
        charge=0,
        multiplicity=1,
    )

    output_df = pdb2df(output_path)
    assert output_path.exists()
    assert len(output_df) == len(pdb2df(reference_path))
    assert list(output_df["ATOM_NAME"]) == list(pdb2df(reference_path)["ATOM_NAME"])
    assert result.validation["hydrogen_optimisation_requested"] is True
    assert result.validation["hydrogen_optimisation_completed"] is True
    for key in (
        "hydrogen_optimisation_workdir",
        "hydrogen_optimisation_staged_pdb",
        "hydrogen_optimisation_input",
        "hydrogen_optimisation_stdout",
        "hydrogen_optimisation_stderr",
        "hydrogen_optimisation_xyz",
        "hydrogen_optimisation_labelled_pdb",
    ):
        assert Path(result.validation[key]).exists(), key


def test_optimisation_requires_all_orca_arguments(tmp_path):
    with pytest.raises(ValueError, match="requires orcadir, charge, and multiplicity"):
        reindex(
            FIXTURES / "palmitic_acid_reference.pdb",
            FIXTURES / "palmitic_acid_target_02.pdb",
            tmp_path / "missing_args.pdb",
            optimise=True,
        )
