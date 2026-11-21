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


def model_cif2df(path: str | Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Read a *model* CIF -- one with an ``_atom_site`` loop -- into the atom frame.

    Separate from :func:`cif2df` on purpose, because the two files answer
    different questions. A component CIF states what a molecule *is*: one
    block, one component, idealised coordinates, and a row order that is
    itself the answer. A model CIF states where atoms *are*: many chains, real
    coordinates, and two parallel numbering schemes. Reading them with one
    function would mean one set of refusals guarding two sets of assumptions.

    **`label_` is preferred over `auth_` wherever both exist.** The label
    scheme is the file's own, contiguous and authored by whatever wrote it;
    the auth scheme is whatever a depositor chose and may repeat, skip or
    carry insertion codes. The reindexer matches on graph structure and writes
    back by row, so what it needs is a numbering that cannot collide -- not
    one that matches somebody's paper. Where `label_` is absent the `auth_`
    column is used and a warning says so.

    Read through biotite rather than this module's own CIF parser: an
    `_atom_site` loop brings auth/label resolution, insertion codes, altlocs
    and multi-model files with it, and a hand-rolled reader of those is a
    silent-renumbering bug waiting to happen. That is the exact failure this
    package exists to prevent.
    """
    import biotite.structure.io.pdbx as pdbx

    path = Path(path)
    logger = logger or _LOG
    try:
        cif = pdbx.CIFFile.read(str(path))
        block = cif.block
    except Exception as problem:
        # biotite's own errors name neither the file nor what was wrong with
        # it -- "Failed to deserialize category 'atom_site'" on an empty loop,
        # for instance. This package's contract is that a CIF which cannot be
        # read raises *naming why*, so the original is kept as the cause and
        # the path is put in front of it.
        raise ValueError(f"{path}: could not be read as a model CIF: {problem}") from problem
    if "atom_site" not in block:
        raise ValueError(
            f"{path} has no _atom_site loop, so it is not a model. Component "
            f"definitions are read by cif2df."
        )
    try:
        # biotite deserialises lazily, so a malformed or empty loop raises
        # here rather than at read(), and says only "Failed to deserialize
        # category 'atom_site'".
        site = block["atom_site"]
        _ = site.row_count
    except Exception as problem:
        raise ValueError(
            f"{path}: its _atom_site loop could not be read: {problem}"
        ) from problem

    def column(label: str, auth: str, *, required: bool = True):
        if label in site:
            return site[label].as_array(str)
        if auth in site:
            logger.warning(
                "%s has no _atom_site.%s; falling back to %s, which a depositor "
                "may have renumbered.", path, label, auth,
            )
            return site[auth].as_array(str)
        if required:
            raise ValueError(f"{path}: _atom_site has neither {label} nor {auth}")
        return None

    group = (
        site["group_PDB"].as_array(str) if "group_PDB" in site
        else np.array(["ATOM"] * site.row_count)
    )
    atom_names = column("label_atom_id", "auth_atom_id")
    res_names = column("label_comp_id", "auth_comp_id")
    chains = column("label_asym_id", "auth_asym_id")
    res_ids = column("label_seq_id", "auth_seq_id")
    elements = (
        site["type_symbol"].as_array(str) if "type_symbol" in site
        else np.array([""] * site.row_count)
    )
    for axis in ("Cartn_x", "Cartn_y", "Cartn_z"):
        if axis not in site:
            raise ValueError(f"{path}: _atom_site has no {axis}")
    xs = site["Cartn_x"].as_array(float)
    ys = site["Cartn_y"].as_array(float)
    zs = site["Cartn_z"].as_array(float)

    models = (
        site["pdbx_PDB_model_num"].as_array(str) if "pdbx_PDB_model_num" in site
        else None
    )
    keep = np.ones(site.row_count, dtype=bool)
    if models is not None:
        distinct = [m for m in dict.fromkeys(models.tolist())]
        if len(distinct) > 1:
            # Refused rather than silently taking the first: which model a
            # caller meant is not this reader's guess to make, and reindexing
            # the wrong one would renumber against the wrong geometry.
            raise ValueError(
                f"{path} holds {len(distinct)} models "
                f"(pdbx_PDB_model_num {', '.join(distinct[:4])}...). Split it "
                f"before reindexing; this reader will not choose one."
            )

    rows = []
    for index in range(site.row_count):
        if not keep[index]:
            continue
        raw_res = str(res_ids[index]).strip()
        if raw_res in _CIF_MISSING or raw_res == "":
            # A ligand's label_seq_id is "." by the dictionary. Numbering it 0
            # keeps the column an integer without inventing a residue index
            # that could collide with a real one.
            residue = 0
        else:
            residue = int(float(raw_res))
        element = str(elements[index]).strip()
        name = str(atom_names[index]).strip()
        rows.append({
            "ATOM": str(group[index]).strip() or "ATOM",
            "ATOM_ID": index + 1,
            "ATOM_NAME": name,
            "RES_NAME": str(res_names[index]).strip(),
            "CHAIN_ID": str(chains[index]).strip(),
            "RES_ID": residue,
            "X": float(xs[index]),
            "Y": float(ys[index]),
            "Z": float(zs[index]),
            "OCCUPANCY": 1.0,
            "BETAFACTOR": 0.0,
            "ELEMENT": _normalise_element(element or _element_from_atom_name(name)),
        })
    if not rows:
        raise ValueError(f"{path}: _atom_site is empty")
    return pd.DataFrame(rows, columns=PDB_COLUMNS)


def df2cif(
    df: pd.DataFrame,
    path: str | Path,
    logger: Optional[logging.Logger] = None,
    block_name: str = "reindexed",
) -> Optional[Path]:
    """Write the atom frame as a model CIF, mirroring :func:`df2pdb`.

    Written directly rather than through biotite because what matters here is
    that **row order is preserved exactly**: the reindexed order is the whole
    output of this package, and a toolkit free to sort by chain or renumber on
    write would quietly undo it. Reading needs a toolkit's auth/label
    knowledge; writing needs only that nothing is reordered.

    Both numbering schemes are written with the same values, so a reader
    preferring either gets what this package decided.

    **Coordinates keep five decimals, not PDB's three.** Three is a
    column-width limit of the PDB format, not a statement about precision, and
    writing it here would re-import a loss this whole phase exists to remove --
    a 0.0005 A round-trip drift on every atom, for nothing. Five matches what
    Boltz2 and the wwPDB write and round-trips their files exactly.
    """
    path = Path(path)
    logger = logger or _LOG
    if not isinstance(df, pd.DataFrame) or df.empty:
        logger.error("Cannot write an empty or non-DataFrame CIF")
        return None
    missing = set(PDB_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")

    def quoted(value: str) -> str:
        text = str(value).strip()
        if text == "":
            return "."
        return f"'{text}'" if any(c.isspace() or c == "'" for c in text) else text

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"data_{block_name}\n#\nloop_\n")
        for tag in (
            "group_PDB", "id", "type_symbol", "label_atom_id", "label_comp_id",
            "label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z",
            "occupancy", "B_iso_or_equiv", "auth_comp_id", "auth_asym_id",
            "auth_seq_id", "auth_atom_id", "pdbx_PDB_model_num",
        ):
            handle.write(f"_atom_site.{tag}\n")
        for index, row in df.reset_index(drop=True).iterrows():
            serial = int(row["ATOM_ID"]) if pd.notna(row["ATOM_ID"]) else index + 1
            name = quoted(row["ATOM_NAME"])
            comp = quoted(row["RES_NAME"])
            asym = quoted(row["CHAIN_ID"])
            seq = int(row["RES_ID"])
            handle.write(
                f"{str(row['ATOM']).strip() or 'ATOM'} {serial} "
                f"{_normalise_element(row['ELEMENT']) or '.'} {name} {comp} "
                f"{asym} {seq} "
                f"{float(row['X']):.5f} {float(row['Y']):.5f} {float(row['Z']):.5f} "
                f"{float(row['OCCUPANCY']):.2f} {float(row['BETAFACTOR']):.2f} "
                f"{comp} {asym} {seq} {name} 1\n"
            )
        handle.write("#\n")
    return path if path.is_file() else None


def read_structure(path: str | Path, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Read a PDB, component CIF or model CIF into the package's atom DataFrame.

    The format is chosen by suffix, and a ``.cif`` is then chosen **by its
    contents**: a block carrying ``_atom_site`` is a model and goes to
    :func:`model_cif2df`, anything else is a component definition and goes to
    :func:`cif2df`. Both extensions are ``.cif`` and the caller usually does
    not know which it holds -- a prediction and a molecule definition arrive
    through the same argument -- so asking the file is the only answer that
    cannot be got wrong by a caller.

    A PDB that cannot be read returns ``None`` as :func:`pdb2df` does; a CIF
    that cannot be read raises, naming why.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return pdb2df(path, logger=logger)
    if suffix == ".cif":
        return model_cif2df(path, logger=logger) if cif_is_model(path) else cif2df(path, logger=logger)
    raise ValueError(f"Unsupported structure format {suffix!r} (expected .pdb or .cif): {path}")


def cif_is_model(path: str | Path) -> bool:
    """Whether this CIF carries an ``_atom_site`` loop, so holds coordinates.

    Read with this package's own parser rather than biotite: deciding *which*
    reader to use must not depend on the reader being decided about, and a
    component definition is not something biotite's model reader should be
    asked to open just to find out it is the wrong kind.
    """
    text = Path(path).read_text(errors="replace")
    return any("_atom_site" in _block_categories(block) for block in _parse_cif(text))


def write_structure(
    df: pd.DataFrame,
    path: str | Path,
    logger: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """Write the atom frame in whichever format ``path`` names.

    The output format follows the caller's filename rather than a flag, so a
    pipeline that wants CIF out asks for it the same way it asks for anything
    else: by saying where to put it.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return df2pdb(df, path, logger=logger)
    if suffix == ".cif":
        return df2cif(df, path, logger=logger)
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
