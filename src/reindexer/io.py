"""Small PDB, XYZ and component-CIF readers/writers used by the reindexer.

The reindexer intentionally owns the limited file-format functionality it
needs.  This keeps it independent from the parent workflow and from a
project-specific structure utility.

The CIF reader reads one kind of CIF only: a *component definition*, whose
atoms are stated in ``_chem_comp_atom`` and which has no ``_atom_site`` loop.
See :func:`cif2df`.
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


_LOG = logging.getLogger(__name__)

#: Residue names written as ATOM rather than HETATM records. A Cbeta-capped
#: amino-acid fragment is a non-polymer *definition*, but what the reindexer
#: writes from it is a residue in a structure, and readers that type a chain
#: of HETATM records as non-polymer will not treat it as a chain. Mirrors
#: ``prometheozyme.common.frames.AMINO_ACID_RESIDUE_NAMES``, which this
#: package deliberately does not import.
AMINO_ACID_RESIDUE_NAMES = frozenset({
    "TRP", "HIS", "TYR", "PHE", "ASN", "ASP", "GLN", "GLU",
    "CYS", "SER", "THR", "LEU", "VAL", "ILE", "MET", "LYS",
    "ARG", "PRO", "ALA", "GLY", "UNK", "MSK",
})

_CIF_MISSING = {"?", "."}
_CIF_IDEAL = tuple(f"pdbx_model_cartn_{dim}_ideal" for dim in "xyz")
_CIF_MODEL = tuple(f"model_cartn_{dim}" for dim in "xyz")


def _cif_tokens(text: str):
    """Yield ``(value, quoted)`` for every CIF token, comments removed.

    ``quoted`` matters because only an *unquoted* ``_tag``, ``loop_`` or
    ``data_`` is syntax; the same text in quotes is a value.
    """
    lines = text.splitlines()
    line_index = 0
    while line_index < len(lines):
        line = lines[line_index]
        line_index += 1
        if line.startswith(";"):
            # Semicolon text field: runs to the next line starting with ';'.
            field = [line[1:]]
            while line_index < len(lines) and not lines[line_index].startswith(";"):
                field.append(lines[line_index])
                line_index += 1
            if line_index >= len(lines):
                raise ValueError("Unterminated semicolon text field")
            line_index += 1
            yield "\n".join(field).strip(), True
            continue
        position, length = 0, len(line)
        while position < length:
            character = line[position]
            if character.isspace():
                position += 1
            elif character == "#":
                break
            elif character in "'\"":
                # A quote closes only where it is followed by whitespace or
                # the end of the line, so C1'-style names survive in "...".
                end = position + 1
                while end < length and not (
                    line[end] == character and (end + 1 == length or line[end + 1].isspace())
                ):
                    end += 1
                if end >= length:
                    raise ValueError(f"Unterminated quoted value: {line!r}")
                yield line[position + 1:end], True
                position = end + 1
            else:
                end = position
                while end < length and not line[end].isspace():
                    end += 1
                yield line[position:end], False
                position = end


def _is_cif_syntax(token: str, quoted: bool) -> bool:
    if quoted:
        return False
    lower = token.lower()
    return (
        lower.startswith(("_", "data_", "save_"))
        or lower in {"loop_", "stop_", "global_"}
    )


def _parse_cif(text: str) -> list[dict]:
    """Parse CIF text into data blocks of single items and loops.

    Each block is ``{"name", "items": {tag: value}, "loops": [(tags, rows)]}``.
    Tags are lower-cased, since CIF tags are case-insensitive; values are not.
    """
    tokens = list(_cif_tokens(text))
    blocks: list[dict] = []
    index = 0
    while index < len(tokens):
        token, quoted = tokens[index]
        lower = token.lower()
        if not quoted and lower.startswith("data_"):
            blocks.append({"name": token[5:], "items": {}, "loops": []})
            index += 1
            continue
        if not blocks:
            raise ValueError(f"CIF content before any data_ block: {token!r}")
        block = blocks[-1]
        if not quoted and lower == "loop_":
            index += 1
            tags = []
            while index < len(tokens) and not tokens[index][1] and tokens[index][0].startswith("_"):
                tags.append(tokens[index][0].lower())
                index += 1
            if not tags:
                raise ValueError("CIF loop_ with no tags")
            values = []
            while index < len(tokens) and not _is_cif_syntax(*tokens[index]):
                values.append(tokens[index][0])
                index += 1
            if len(values) % len(tags):
                raise ValueError(
                    f"CIF loop over {tags[0].split('.')[0]} has {len(values)} values, "
                    f"not a multiple of its {len(tags)} columns"
                )
            rows = [values[start:start + len(tags)] for start in range(0, len(values), len(tags))]
            block["loops"].append((tags, rows))
        elif not quoted and token.startswith("_"):
            if index + 1 >= len(tokens) or _is_cif_syntax(*tokens[index + 1]):
                raise ValueError(f"CIF item {token} has no value")
            block["items"][lower] = tokens[index + 1][0]
            index += 2
        else:
            raise ValueError(f"Unexpected CIF token {token!r}")
    return blocks


def _block_categories(block: dict) -> set[str]:
    tags = list(block["items"]) + [tag for tags, _ in block["loops"] for tag in tags]
    return {tag.split(".", 1)[0] for tag in tags}


def _coordinates(columns, rows, names, fields, label, path):
    """One coordinate set as an (n, 3) array, or the reason it is unusable."""
    missing = [field for field in fields if field not in columns]
    if missing:
        return None, f"{path.name} has no {label} coordinate columns ({', '.join(missing)})"
    unplaced = [
        names[row_index]
        for row_index, row in enumerate(rows)
        if any(row[columns[field]] in _CIF_MISSING for field in fields)
    ]
    if unplaced:
        return None, f"{path.name} leaves atoms without {label} coordinates: {', '.join(unplaced)}"
    try:
        array = np.asarray(
            [[float(row[columns[field]]) for field in fields] for row in rows], dtype=float
        )
    except ValueError as exc:
        raise ValueError(f"{path.name} has a non-numeric {label} coordinate: {exc}") from exc
    return array, None


def cif2df(path: str | Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a component-definition CIF's ``_chem_comp_atom`` loop.

    A component CIF *defines* a molecule rather than modelling one. Its atoms
    are the rows of ``_chem_comp_atom``, and an atom is addressed by its
    0-based position there; downstream graph configs, assembly components and
    ORCA inputs all use that same number. Row order is therefore the contract,
    and the returned DataFrame keeps it exactly: row ``i`` is file row ``i``.

    Columns are found by their header, never by position, because the two
    flavours in use differ in width: an 8-column library fragment and a
    21-column wwPDB dictionary entry. Coordinates are the *ideal* set
    (``pdbx_model_Cartn_*_ideal``); in a dictionary entry the ``model_Cartn_*``
    set is a different, experimental geometry and is read only if the ideal
    set is absent or incomplete, which is logged. A definition that leaves an
    atom unplaced in both is refused.

    Refused rather than guessed at, because each would silently repoint
    atoms: a file with an ``_atom_site`` loop (a model, with a second
    numbering — not this reader's kind); more than one component; a
    ``pdbx_ordinal`` that disagrees with row order; duplicate atom names.

    ``ATOM_ID`` is the 1-based position, and chain ``A`` / residue ``1`` are
    invented, because a definition states neither. ``df.attrs`` carries
    ``comp_id``, per-atom ``charges`` and the declared ``formal_charge``
    (``_chem_comp.pdbx_formal_charge``, ``None`` if absent); a disagreement
    between the declared charge and the atom charges is logged, not resolved.

    Raises ``FileNotFoundError`` or ``ValueError`` naming the problem; unlike
    :func:`pdb2df` it never returns ``None``.
    """
    log = logger or _LOG
    path = Path(path)
    if path.suffix.lower() != ".cif":
        raise ValueError(f"Not a .cif file: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"CIF file does not exist: {path}")

    blocks = _parse_cif(path.read_text())
    modelled = [block["name"] for block in blocks if "_atom_site" in _block_categories(block)]
    if modelled:
        raise ValueError(
            f"{path} has an _atom_site loop (data_{modelled[0]}), so it models a "
            "structure rather than defining a component. The reindexer reads only "
            "component definitions from CIF, whose atoms are _chem_comp_atom alone."
        )
    found = [
        (block, tags, rows)
        for block in blocks
        for tags, rows in block["loops"]
        if tags[0].startswith("_chem_comp_atom.")
    ]
    if not found:
        raise ValueError(f"{path} has no _chem_comp_atom loop")
    if len(found) > 1:
        raise ValueError(
            f"{path} has {len(found)} _chem_comp_atom loops; a component CIF defines one molecule"
        )
    block, tags, rows = found[0]
    if not rows:
        raise ValueError(f"{path} states no atoms in its _chem_comp_atom loop")
    columns = {tag.split(".", 1)[1]: position for position, tag in enumerate(tags)}

    required = ("comp_id", "atom_id", "type_symbol")
    missing = [column for column in required if column not in columns]
    if missing:
        raise ValueError(f"{path} _chem_comp_atom lacks {', '.join(missing)}")

    names = [row[columns["atom_id"]] for row in rows]
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"{path} names more than one atom {', '.join(duplicated)}")

    comp_ids = {row[columns["comp_id"]] for row in rows}
    if len(comp_ids) != 1:
        raise ValueError(f"{path} defines more than one component: {', '.join(sorted(comp_ids))}")
    comp_id = comp_ids.pop()

    if "pdbx_ordinal" in columns:
        ordinals = [row[columns["pdbx_ordinal"]] for row in rows]
        expected = [str(position) for position in range(1, len(rows) + 1)]
        if ordinals != expected:
            first = next(i for i, (a, b) in enumerate(zip(ordinals, expected)) if a != b)
            raise ValueError(
                f"{path} pdbx_ordinal disagrees with row order at row {first} "
                f"({names[first]}: ordinal {ordinals[first]}, position {first + 1}). "
                "Atoms are addressed by row position; refusing a second numbering."
            )

    coordinates, ideal_problem = _coordinates(columns, rows, names, _CIF_IDEAL, "ideal", path)
    if coordinates is None:
        coordinates, model_problem = _coordinates(columns, rows, names, _CIF_MODEL, "model", path)
        if coordinates is None:
            raise ValueError(f"{ideal_problem}; and {model_problem}")
        log.warning("%s; using the model_Cartn_* coordinates instead", ideal_problem)
    elif all(field in columns for field in _CIF_MODEL):
        log.debug(
            "%s: ignoring the model_Cartn_* coordinate set; the ideal set is the definition's",
            path.name,
        )

    elements = [_normalise_element(row[columns["type_symbol"]]) for row in rows]
    empty = [names[i] for i, element in enumerate(elements) if not element]
    if empty:
        raise ValueError(f"{path} gives no type_symbol for {', '.join(empty)}")

    charges = []
    for row, name in zip(rows, names):
        value = row[columns["charge"]] if "charge" in columns else "?"
        try:
            charges.append(0 if value in _CIF_MISSING else int(value))
        except ValueError as exc:
            raise ValueError(f"{path} atom {name} has a non-integer charge {value!r}") from exc

    formal_value = block["items"].get("_chem_comp.pdbx_formal_charge")
    formal_charge = None
    if formal_value is not None and formal_value not in _CIF_MISSING:
        try:
            formal_charge = int(formal_value)
        except ValueError as exc:
            raise ValueError(f"{path} pdbx_formal_charge is not an integer: {formal_value!r}") from exc
        if formal_charge != sum(charges):
            log.warning(
                "%s declares pdbx_formal_charge %d but its atom charges sum to %d; "
                "the declared charge is authoritative",
                path.name, formal_charge, sum(charges),
            )

    count = len(rows)
    record = "ATOM" if comp_id.strip().upper() in AMINO_ACID_RESIDUE_NAMES else "HETATM"
    df = pd.DataFrame(
        {
            "ATOM": [record] * count,
            "ATOM_ID": list(range(1, count + 1)),
            "ATOM_NAME": names,
            "RES_NAME": [comp_id] * count,
            "CHAIN_ID": ["A"] * count,
            "RES_ID": [1] * count,
            "X": coordinates[:, 0],
            "Y": coordinates[:, 1],
            "Z": coordinates[:, 2],
            "OCCUPANCY": [1.0] * count,
            "BETAFACTOR": [0.0] * count,
            "ELEMENT": elements,
        },
        columns=PDB_COLUMNS,
    )
    df.attrs.update({"comp_id": comp_id, "charges": charges, "formal_charge": formal_charge})
    return df


def read_structure(path: str | Path, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Read a PDB or component CIF into the package's atom DataFrame.

    Chosen by suffix. A PDB that cannot be read returns ``None`` as
    :func:`pdb2df` does; a CIF that cannot be read raises, naming why.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return pdb2df(path, logger=logger)
    if suffix == ".cif":
        return cif2df(path, logger=logger)
    raise ValueError(f"Unsupported structure format {suffix!r} (expected .pdb or .cif): {path}")


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
