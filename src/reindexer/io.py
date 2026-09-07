"""Small PDB and XYZ readers/writers used by the reindexing package.

The reindexer intentionally owns the limited file-format functionality it
needs.  This keeps it independent from the parent workflow and from a
project-specific structure utility.
"""

from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd


PDB_COLUMNS = (
    "ATOM",
    "ATOM_ID",
    "ATOM_NAME",
    "RES_NAME",
    "CHAIN_ID",
    "RES_ID",
    "X",
    "Y",
    "Z",
    "OCCUPANCY",
    "BETAFACTOR",
    "ELEMENT",
)


def _normalise_element(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return text[0].upper() + text[1:].lower()


def _element_from_atom_name(atom_name: str) -> str:
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        return ""
    if len(letters) > 1 and letters[:2].title() in {"Cl", "Br"}:
        return letters[:2].title()
    return letters[0].upper()


def pdb2df(path: str | Path, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Read ATOM/HETATM records from a PDB file into a DataFrame.

    The parser follows the fixed-width PDB fields used by the package and
    ignores headers, connectivity records, and model delimiters.
    """
    path = Path(path)
    if path.suffix.lower() != ".pdb" or not path.is_file():
        if logger:
            logger.error("PDB file does not exist or has the wrong suffix: %s", path)
        return None

    rows = []
    try:
        with path.open() as handle:
            for line_number, line in enumerate(handle, start=1):
                record = line[:6].strip().upper()
                if record not in {"ATOM", "HETATM"}:
                    continue
                try:
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() or _element_from_atom_name(atom_name)
                    rows.append(
                        {
                            "ATOM": record,
                            "ATOM_ID": int(line[6:11].strip()),
                            "ATOM_NAME": atom_name,
                            "RES_NAME": line[17:20].strip(),
                            "CHAIN_ID": line[21:22].strip(),
                            "RES_ID": int(line[22:26].strip()),
                            "X": float(line[30:38].strip()),
                            "Y": float(line[38:46].strip()),
                            "Z": float(line[46:54].strip()),
                            "OCCUPANCY": float(line[54:60].strip() or 1.0),
                            "BETAFACTOR": float(line[60:66].strip() or 0.0),
                            "ELEMENT": _normalise_element(element),
                        }
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Malformed PDB atom record on line {line_number}: {line.rstrip()}") from exc
    except Exception as exc:
        if logger:
            logger.exception("Failed to read PDB file %s: %s", path, exc)
            return None
        raise

    if not rows:
        return None
    return pd.DataFrame(rows, columns=PDB_COLUMNS)


def df2pdb(
    df: pd.DataFrame,
    path: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write the package's PDB DataFrame representation to disk."""
    path = Path(path)
    if not isinstance(df, pd.DataFrame) or df.empty:
        if logger:
            logger.error("Cannot write an empty or non-DataFrame PDB")
        return None
    missing = set(PDB_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing PDB columns: {sorted(missing)}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, row in df.reset_index(drop=True).iterrows():
            record = str(row["ATOM"]).strip() or "ATOM"
            serial = int(row["ATOM_ID"]) if pd.notna(row["ATOM_ID"]) else index + 1
            atom_name = str(row["ATOM_NAME"]).strip()
            residue = str(row["RES_NAME"]).strip()[:3]
            chain = str(row["CHAIN_ID"]).strip()[:1]
            residue_id = int(row["RES_ID"])
            element = _normalise_element(row["ELEMENT"])
            handle.write(
                f"{record:<6}{serial:5d} {atom_name:>4} {residue:>3} {chain:1}"
                f"{residue_id:4d}    {float(row['X']):8.3f}{float(row['Y']):8.3f}"
                f"{float(row['Z']):8.3f}{float(row['OCCUPANCY']):6.2f}"
                f"{float(row['BETAFACTOR']):6.2f}          {element:>2}\n"
            )
        handle.write("END\n")
    return path if path.is_file() else None


def read_xyz(path: str | Path) -> tuple[int, str, np.ndarray, list[str]]:
    """Read one strict XYZ structure."""
    path = Path(path)
    with path.open() as handle:
        first_line = handle.readline()
        if not first_line:
            raise ValueError(f"XYZ file is empty: {path}")
        try:
            atom_count = int(first_line.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid XYZ atom count in {path}: {first_line!r}") from exc
        comment = handle.readline()
        if comment == "":
            raise ValueError(f"XYZ file ended before comment line: {path}")

        elements = []
        coordinates = []
        for atom_index in range(atom_count):
            line = handle.readline()
            if line == "":
                raise ValueError(f"XYZ file ended early: expected {atom_count} atoms, got {atom_index}")
            fields = line.split()
            if len(fields) < 4:
                raise ValueError(f"Malformed XYZ coordinate row {atom_index + 3}: {line!r}")
            try:
                coordinates.append([float(fields[1]), float(fields[2]), float(fields[3])])
            except ValueError as exc:
                raise ValueError(f"Invalid XYZ coordinates on row {atom_index + 3}: {line!r}") from exc
            elements.append(fields[0])

    array = np.asarray(coordinates, dtype=float)
    if array.shape != (atom_count, 3):
        raise ValueError(f"XYZ coordinate shape mismatch in {path}: got {array.shape}")
    return atom_count, comment.rstrip("\n"), array, elements


def xyz2df(
    path: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[tuple[int, str, pd.DataFrame]]:
    """Read an XYZ file and return its count, comment, and atom DataFrame."""
    path = Path(path)
    if path.suffix.lower() != ".xyz" or not path.is_file():
        if logger:
            logger.error("XYZ file does not exist or has the wrong suffix: %s", path)
        return None
    try:
        atom_count, comment, coordinates, elements = read_xyz(path)
    except Exception:
        if logger:
            logger.exception("Failed to read XYZ file %s", path)
            return None
        raise
    return atom_count, comment, pd.DataFrame(
        {
            "ELEMENT": elements,
            "X": coordinates[:, 0],
            "Y": coordinates[:, 1],
            "Z": coordinates[:, 2],
        }
    )
