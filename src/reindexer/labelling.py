"""Apply ORCA XYZ coordinates to reference PDB atom labels and ordering."""

from pathlib import Path

import numpy as np

from .io import df2pdb, pdb2df, read_xyz


class LabellingError(ValueError):
    """Raised when an XYZ file cannot be safely labelled with a PDB."""


def _normalise_element(value: object) -> str:
    element = str(value).strip()
    if not element:
        return element
    return element[0].upper() + element[1:].lower()


def label_xyz_with_reference(
    reference_pdb: str | Path,
    xyz_file: str | Path,
    output_pdb: str | Path,
) -> Path:
    """Write XYZ coordinates using the reference PDB's atom identity/order."""
    reference_pdb = Path(reference_pdb)
    xyz_file = Path(xyz_file)
    output_pdb = Path(output_pdb)

    reference_df = pdb2df(reference_pdb)
    if reference_df is None or reference_df.empty:
        raise LabellingError(f"Reference PDB contains no atom records: {reference_pdb}")

    atom_count, _, coordinates, elements = read_xyz(xyz_file)
    if atom_count != len(reference_df):
        raise LabellingError(
            f"Atom count mismatch: reference PDB has {len(reference_df)}, "
            f"XYZ has {atom_count}"
        )

    reference_elements = [
        _normalise_element(value) for value in reference_df["ELEMENT"]
    ]
    xyz_elements = [_normalise_element(value) for value in elements]
    if reference_elements != xyz_elements:
        mismatch = next(
            (
                index,
                reference_element,
                xyz_element,
            )
            for index, (reference_element, xyz_element) in enumerate(
                zip(reference_elements, xyz_elements)
            )
            if reference_element != xyz_element
        )
        raise LabellingError(
            "PDB/XYZ element order mismatch at atom index "
            f"{mismatch[0]}: PDB={mismatch[1]!r}, XYZ={mismatch[2]!r}"
        )

    output_df = reference_df.copy()
    output_df[["X", "Y", "Z"]] = np.round(coordinates, 3)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    df2pdb(output_df, output_pdb)
    if not output_pdb.is_file():
        raise LabellingError(f"PDB writer did not create output: {output_pdb}")
    return output_pdb
