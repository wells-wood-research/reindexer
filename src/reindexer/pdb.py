"""Reference-ordered PDB reindexing with target-coordinate preservation.

The PDB document model in this module deliberately remains separate from the
RDKit molecule.  RDKit is used for chemistry and atom mapping; the document
model is used to preserve PDB fields and to write records in reference order.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .errors import (
    AmbiguousAtomMappingError,
    ChemicalMappingError,
    HydrogenReconciliationError,
    PDBFormatError,
    UnsupportedPDBFeatureError,
)


ATOM_RECORDS = {"ATOM", "HETATM"}
SERIAL_POLICIES = {"reference", "target"}
_STRUCTURAL_RECORDS = {"ANISOU", "CONECT", "END", "ENDMDL", "MODEL", "TER"}


@dataclass(frozen=True)
class PDBAtom:
    """A parsed PDB atom record and its source formatting."""

    record_name: str
    serial: int
    atom_name: str
    atom_name_field: str
    alt_loc: str
    residue_name: str
    residue_name_field: str
    chain_id: str
    residue_sequence: int
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: float
    b_factor: float
    element: str
    element_field: str
    charge: str
    charge_field: str
    raw_line: str
    line_number: int

    @property
    def residue_key(self) -> Tuple[str, str, int, str]:
        return (
            self.chain_id,
            self.residue_name.upper(),
            self.residue_sequence,
            self.insertion_code,
        )


@dataclass(frozen=True)
class PDBDocument:
    """Parsed PDB data needed for safe reordering and round-tripping."""

    path: Path
    atoms: Tuple[PDBAtom, ...]
    metadata_lines: Tuple[str, ...]
    anisou_by_serial: Mapping[int, str]
    had_ter_records: bool

    @property
    def atoms_by_serial(self) -> Mapping[int, PDBAtom]:
        return {atom.serial: atom for atom in self.atoms}


@dataclass(frozen=True)
class ReindexResult:
    """Result and audit information from :func:`reindex_pdb`."""

    output_path: Path
    serial_policy: str
    reference_to_target: Mapping[int, int]
    generated_hydrogens: Tuple[int, ...]
    removed_target_hydrogens: Tuple[int, ...]


def default_output_path(target: str | Path) -> Path:
    """Return the default sibling output path for a target PDB."""

    target_path = Path(target)
    return target_path.with_name(f"{target_path.stem}_reindexed{target_path.suffix}")


def _field(line: str, start: int, end: int) -> str:
    return line[start:end] if len(line) >= start else ""


def _parse_int(value: str, description: str, line_number: int) -> int:
    try:
        return int(value.strip())
    except ValueError as exc:
        raise PDBFormatError(
            f"Line {line_number}: invalid {description} value {value!r}."
        ) from exc


def _parse_float(value: str, description: str, line_number: int) -> float:
    try:
        result = float(value.strip())
    except ValueError as exc:
        raise PDBFormatError(
            f"Line {line_number}: invalid {description} value {value!r}."
        ) from exc
    if result != result or result in (float("inf"), float("-inf")):
        raise PDBFormatError(f"Line {line_number}: {description} must be finite.")
    return result


def _infer_element(atom_name: str) -> str:
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        return ""
    if len(letters) >= 2 and letters[:2].upper() in {
        "CL",
        "BR",
        "SI",
        "NA",
        "MG",
        "FE",
        "ZN",
        "CA",
        "CU",
        "MN",
        "CO",
        "NI",
    }:
        return letters[:2].upper()
    return letters[0].upper()


def _parse_atom(line: str, line_number: int) -> PDBAtom:
    if len(line) < 54:
        raise PDBFormatError(
            f"Line {line_number}: atom record is shorter than the coordinate fields."
        )

    atom_name_field = _field(line, 12, 16)
    residue_name_field = _field(line, 17, 20)
    element_field = _field(line, 76, 78)
    charge_field = _field(line, 78, 80)
    atom_name = atom_name_field.strip()
    element = element_field.strip().upper() or _infer_element(atom_name)
    if not element:
        raise PDBFormatError(f"Line {line_number}: atom has no element or usable name.")

    return PDBAtom(
        record_name=_field(line, 0, 6).strip(),
        serial=_parse_int(_field(line, 6, 11), "atom serial", line_number),
        atom_name=atom_name,
        atom_name_field=atom_name_field,
        alt_loc=_field(line, 16, 17).strip(),
        residue_name=residue_name_field.strip(),
        residue_name_field=residue_name_field,
        chain_id=_field(line, 21, 22).strip(),
        residue_sequence=_parse_int(_field(line, 22, 26), "residue sequence", line_number),
        insertion_code=_field(line, 26, 27).strip(),
        x=_parse_float(_field(line, 30, 38), "x coordinate", line_number),
        y=_parse_float(_field(line, 38, 46), "y coordinate", line_number),
        z=_parse_float(_field(line, 46, 54), "z coordinate", line_number),
        occupancy=_parse_float(_field(line, 54, 60), "occupancy", line_number),
        b_factor=_parse_float(_field(line, 60, 66), "B-factor", line_number),
        element=element,
        element_field=element_field,
        charge=charge_field.strip(),
        charge_field=charge_field,
        raw_line=line.rstrip("\r\n"),
        line_number=line_number,
    )


def parse_pdb(path: str | Path) -> PDBDocument:
    """Parse a single-model PDB while retaining compatible raw metadata.

    Alternate-location records and model records are rejected rather than
    silently treating coordinate variants as separate chemical atoms.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"PDB file not found: {source}")

    atoms: List[PDBAtom] = []
    metadata: List[str] = []
    anisou: Dict[int, str] = {}
    seen_serials = set()
    had_ter = False
    model_count = 0

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            record_name = _field(line, 0, 6).strip().upper()
            if record_name in ATOM_RECORDS:
                atom = _parse_atom(line, line_number)
                if atom.serial in seen_serials:
                    raise PDBFormatError(f"Duplicate atom serial {atom.serial}.")
                if atom.alt_loc:
                    raise UnsupportedPDBFeatureError(
                        f"Line {line_number}: alternate location {atom.alt_loc!r} "
                        "is not supported safely."
                    )
                seen_serials.add(atom.serial)
                atoms.append(atom)
            elif record_name == "ANISOU":
                serial = _parse_int(_field(line, 6, 11), "ANISOU atom serial", line_number)
                anisou[serial] = line
            elif record_name == "MODEL":
                model_count += 1
                raise UnsupportedPDBFeatureError("MODEL records are not supported.")
            elif record_name == "TER":
                had_ter = True
            elif record_name not in _STRUCTURAL_RECORDS:
                metadata.append(line)

    if not atoms:
        raise PDBFormatError(f"No ATOM or HETATM records found in {source}.")
    return PDBDocument(source, tuple(atoms), tuple(metadata), anisou, had_ter)


def _rdkit_modules():
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
        from rdkit.Geometry import Point3D  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "PDB reindexing requires the 'rdkit' dependency."
        ) from exc
    return Chem, Point3D


def _load_rdkit_molecule(document: PDBDocument):
    Chem, _ = _rdkit_modules()
    molecule = Chem.MolFromPDBFile(
        str(document.path),
        sanitize=False,
        removeHs=False,
        proximityBonding=True,
    )
    if molecule is None:
        raise ChemicalMappingError(f"RDKit could not parse {document.path}.")
    if molecule.GetNumAtoms() != len(document.atoms):
        raise ChemicalMappingError(
            f"RDKit parsed {molecule.GetNumAtoms()} atoms from {document.path}, "
            f"but the PDB contains {len(document.atoms)} atom records."
        )
    for index, atom in enumerate(molecule.GetAtoms()):
        atom.SetIntProp("_reindexer_pdb_serial", document.atoms[index].serial)
    return molecule


def _heavy_molecule(molecule):
    Chem, _ = _rdkit_modules()
    try:
        heavy = Chem.RemoveHs(molecule, sanitize=False)
        Chem.SanitizeMol(heavy)
    except Exception as exc:
        raise ChemicalMappingError(
            "RDKit could not sanitize the heavy-atom representation. "
            "Check PDB connectivity, valence, and formal charges."
        ) from exc
    return heavy


def _connectivity_view(molecule):
    """Return an RDKit view that matches atom adjacency, not bond order.

    PDB ``CONECT`` records often do not preserve double/aromatic bond order,
    and different hydrogenation states can make RDKit perceive those orders
    differently.  Mapping should therefore use element identity and graph
    connectivity only.  The original reference molecule remains untouched and
    is used later for canonical output connectivity and hydrogenation.
    """

    Chem, _ = _rdkit_modules()
    view = Chem.Mol(molecule)
    for atom in view.GetAtoms():
        # Mapping is intentionally independent of hydrogenation, charge, and
        # aromaticity. Those properties are supplied by the reference later.
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(0)
        atom.SetNoImplicit(True)
        atom.SetIsAromatic(False)
    for bond in view.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return view


def _record_for_serial(document: PDBDocument, serial: int) -> PDBAtom:
    try:
        return document.atoms_by_serial[serial]
    except KeyError as exc:
        raise ChemicalMappingError(
            f"RDKit referred to atom serial {serial}, which is absent from the PDB."
        ) from exc


def _mapping_score(
    reference: PDBDocument,
    target: PDBDocument,
    reference_heavy,
    target_heavy,
    match: Sequence[int],
) -> int:
    score = 0
    for reference_index, target_index in enumerate(match):
        reference_serial = reference_heavy.GetAtomWithIdx(reference_index).GetIntProp(
            "_reindexer_pdb_serial"
        )
        target_serial = target_heavy.GetAtomWithIdx(target_index).GetIntProp(
            "_reindexer_pdb_serial"
        )
        reference_atom = _record_for_serial(reference, reference_serial)
        target_atom = _record_for_serial(target, target_serial)
        if reference_atom.atom_name.upper() == target_atom.atom_name.upper():
            score += 100
        if reference_atom.residue_key == target_atom.residue_key:
            score += 25
        if reference_atom.record_name == target_atom.record_name:
            score += 5
        if reference_atom.element == target_atom.element:
            score += 2
        if reference_atom.charge == target_atom.charge:
            score += 10
    return score


def _map_heavy_atoms(reference: PDBDocument, target: PDBDocument, reference_mol, target_mol):
    reference_heavy = _heavy_molecule(reference_mol)
    target_heavy = _heavy_molecule(target_mol)
    if reference_heavy.GetNumAtoms() != target_heavy.GetNumAtoms():
        raise ChemicalMappingError(
            "Reference and target contain different numbers of heavy atoms. "
            "Only hydrogen count differences can be reconciled automatically."
        )

    query = _connectivity_view(reference_heavy)
    search = _connectivity_view(target_heavy)
    matches = search.GetSubstructMatches(
        query,
        uniquify=False,
        useChirality=True,
        maxMatches=10000,
    )
    if not matches:
        raise ChemicalMappingError(
            "Reference and target heavy-atom graphs are not chemically isomorphic."
        )

    scored = [
        (_mapping_score(reference, target, reference_heavy, target_heavy, match), match)
        for match in matches
    ]
    best_score = max(score for score, _ in scored)
    best = [match for score, match in scored if score == best_score]
    if len(best) != 1:
        raise AmbiguousAtomMappingError(
            f"{len(best)} equally valid heavy-atom mappings remain after using "
            "element, atom-name, and residue metadata."
        )

    match = best[0]
    reference_to_target = {}
    for reference_index, target_index in enumerate(match):
        reference_serial = reference_heavy.GetAtomWithIdx(reference_index).GetIntProp(
            "_reindexer_pdb_serial"
        )
        target_serial = target_heavy.GetAtomWithIdx(target_index).GetIntProp(
            "_reindexer_pdb_serial"
        )
        reference_to_target[reference_serial] = target_serial
    return reference_mol, target_mol, reference_heavy, target_heavy, reference_to_target


def _hydrogen_parent(molecule, serial: int) -> Optional[int]:
    for atom in molecule.GetAtoms():
        if atom.GetIntProp("_reindexer_pdb_serial") != serial:
            continue
        if atom.GetSymbol().upper() != "H":
            return None
        neighbors = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetSymbol().upper() != "H"]
        if len(neighbors) == 1:
            return neighbors[0].GetIntProp("_reindexer_pdb_serial")
        return None
    return None


def _map_hydrogens(
    reference: PDBDocument,
    target: PDBDocument,
    reference_mol,
    target_mol,
    heavy_mapping: Mapping[int, int],
):
    used = set(heavy_mapping.values())
    reference_to_target: Dict[int, int] = dict(heavy_mapping)
    target_hydrogens = [atom for atom in target.atoms if atom.element == "H"]

    for reference_atom in reference.atoms:
        if reference_atom.element != "H":
            continue
        parent = _hydrogen_parent(reference_mol, reference_atom.serial)
        if parent is None or parent not in heavy_mapping:
            raise HydrogenReconciliationError(
                f"Reference hydrogen {reference_atom.serial} has no unique heavy-atom parent."
            )
        target_parent = heavy_mapping[parent]
        candidates = []
        for target_atom in target_hydrogens:
            if target_atom.serial in used:
                continue
            target_h_parent = _hydrogen_parent(target_mol, target_atom.serial)
            score = 0
            if target_h_parent == target_parent:
                score += 50
            if target_atom.atom_name.upper() == reference_atom.atom_name.upper():
                score += 100
            if target_atom.residue_key == reference_atom.residue_key:
                score += 25
            if score:
                candidates.append((score, target_atom.serial))

        if candidates:
            best_score = max(score for score, _ in candidates)
            best = [serial for score, serial in candidates if score == best_score]
            if len(best) != 1:
                raise AmbiguousAtomMappingError(
                    f"Hydrogen {reference_atom.serial} has multiple equally valid target atoms."
                )
            reference_to_target[reference_atom.serial] = best[0]
            used.add(best[0])

    removed = tuple(atom.serial for atom in target_hydrogens if atom.serial not in used)
    return reference_to_target, removed


def _generated_hydrogen_coordinates(
    reference: PDBDocument,
    target: PDBDocument,
    reference_mol,
    reference_heavy,
    mapping,
):
    Chem, Point3D = _rdkit_modules()
    if reference_heavy.GetNumConformers() == 0:
        raise HydrogenReconciliationError("Reference molecule has no usable 3D conformer.")

    conformer = reference_heavy.GetConformer()
    for reference_index in range(reference_heavy.GetNumAtoms()):
        reference_serial = reference_heavy.GetAtomWithIdx(reference_index).GetIntProp(
            "_reindexer_pdb_serial"
        )
        target_serial = mapping[reference_serial]
        target_atom = _record_for_serial(target, target_serial)
        conformer.SetAtomPosition(reference_index, Point3D(target_atom.x, target_atom.y, target_atom.z))

    with_hydrogens = Chem.AddHs(reference_heavy, addCoords=True)
    generated_by_parent: Dict[int, List[Tuple[float, float, float]]] = {}
    added_conformer = with_hydrogens.GetConformer()
    for atom in with_hydrogens.GetAtoms():
        if atom.GetSymbol().upper() != "H" or atom.GetDegree() != 1:
            continue
        parent = atom.GetNeighbors()[0]
        parent_serial = parent.GetIntProp("_reindexer_pdb_serial")
        point = added_conformer.GetAtomPosition(atom.GetIdx())
        generated_by_parent.setdefault(parent_serial, []).append((point.x, point.y, point.z))

    result: Dict[int, Tuple[float, float, float]] = {}
    reference_atoms_by_parent: Dict[int, List[int]] = {}
    for atom in reference.atoms:
        if atom.element != "H":
            continue
        parent = _hydrogen_parent(reference_mol, atom.serial)
        if parent is not None:
            reference_atoms_by_parent.setdefault(parent, []).append(atom.serial)

    for parent, serials in reference_atoms_by_parent.items():
        generated = generated_by_parent.get(parent, [])
        if len(generated) != len(serials):
            raise HydrogenReconciliationError(
                f"RDKit generated {len(generated)} hydrogens for reference atom "
                f"{parent}, but the reference requires {len(serials)}."
            )
        for serial, point in zip(serials, generated):
            result[serial] = point
    return result


def _replace_field(line: str, start: int, end: int, value: object, right: bool = False) -> str:
    width = end - start
    text = str(value)
    if len(text) > width:
        raise PDBFormatError(f"Value {text!r} does not fit PDB columns {start + 1}-{end}.")
    text = text.rjust(width) if right else text.ljust(width)
    padded = line.ljust(end)
    return padded[:start] + text + padded[end:]


def _render_atom(atom: PDBAtom, canonical: PDBAtom, serial: int, point=None) -> str:
    line = atom.raw_line
    line = _replace_field(line, 6, 11, serial, right=True)
    line = _replace_field(line, 12, 16, canonical.atom_name_field)
    line = _replace_field(line, 17, 20, canonical.residue_name_field)
    line = _replace_field(line, 21, 22, canonical.chain_id)
    line = _replace_field(line, 22, 26, canonical.residue_sequence, right=True)
    line = _replace_field(line, 26, 27, canonical.insertion_code)
    line = _replace_field(line, 76, 78, canonical.element_field)
    line = _replace_field(line, 78, 80, canonical.charge_field)
    if point is not None:
        x, y, z = point
        line = _replace_field(line, 30, 38, f"{x:8.3f}")
        line = _replace_field(line, 38, 46, f"{y:8.3f}")
        line = _replace_field(line, 46, 54, f"{z:8.3f}")
    return line.rstrip()


def _render_anisou(line: str, canonical: PDBAtom, serial: int) -> str:
    line = _replace_field(line, 6, 11, serial, right=True)
    line = _replace_field(line, 12, 16, canonical.atom_name_field)
    line = _replace_field(line, 17, 20, canonical.residue_name_field)
    line = _replace_field(line, 21, 22, canonical.chain_id)
    line = _replace_field(line, 22, 26, canonical.residue_sequence, right=True)
    line = _replace_field(line, 26, 27, canonical.insertion_code)
    line = _replace_field(line, 76, 78, canonical.element_field)
    line = _replace_field(line, 78, 80, canonical.charge_field)
    return line.rstrip()


def _render_ter(atom: PDBAtom, serial: int) -> str:
    return (
        f"TER   {serial:5d}      {atom.residue_name_field} {atom.chain_id}"
        f"{atom.residue_sequence:4d}{atom.insertion_code}"
    )


def _conect_lines(reference_mol, serial_by_reference_serial: Mapping[int, int]) -> List[str]:
    lines: List[str] = []
    for atom in reference_mol.GetAtoms():
        reference_serial = atom.GetIntProp("_reindexer_pdb_serial")
        neighbors: List[int] = []
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            other_serial = other.GetIntProp("_reindexer_pdb_serial")
            order = int(round(float(bond.GetBondTypeAsDouble())))
            neighbors.extend([serial_by_reference_serial[other_serial]] * max(order, 1))
        if not neighbors:
            continue
        output_serial = serial_by_reference_serial[reference_serial]
        for offset in range(0, len(neighbors), 4):
            chunk = neighbors[offset : offset + 4]
            lines.append("CONECT" + f"{output_serial:5d}" + "".join(f"{serial:5d}" for serial in chunk))
    return lines


def reindex_pdb(
    reference: str | Path,
    target: str | Path,
    output: str | Path | None = None,
    *,
    serial_policy: str = "reference",
) -> ReindexResult:
    """Write ``target`` in the canonical atom order of ``reference``.

    ``reference`` supplies atom identity, residue identity, elements, charges,
    connectivity, and hydrogenation.  Existing target atom records supply
    coordinates, occupancy, B-factor, and compatible formatting.  The serial
    policy controls whether output serials come from the reference or target;
    missing hydrogens receive new serials under the target policy.
    """

    if Path(reference).suffix.lower() != ".pdb" or Path(target).suffix.lower() != ".pdb":
        raise ValueError("reindex_pdb requires a reference PDB and a target PDB.")
    if serial_policy not in SERIAL_POLICIES:
        raise ValueError(
            f"serial_policy must be one of {sorted(SERIAL_POLICIES)}, got {serial_policy!r}."
        )

    reference_document = parse_pdb(reference)
    target_document = parse_pdb(target)
    reference_mol = _load_rdkit_molecule(reference_document)
    target_mol = _load_rdkit_molecule(target_document)
    (
        reference_mol,
        target_mol,
        reference_heavy,
        _target_heavy,
        heavy_mapping,
    ) = _map_heavy_atoms(reference_document, target_document, reference_mol, target_mol)

    mapping, removed_hydrogens = _map_hydrogens(
        reference_document,
        target_document,
        reference_mol,
        target_mol,
        heavy_mapping,
    )
    generated_coordinates = _generated_hydrogen_coordinates(
        reference_document,
        target_document,
        reference_mol,
        reference_heavy,
        heavy_mapping,
    )

    target_by_serial = target_document.atoms_by_serial
    used_target_serials = set(mapping.values())
    next_serial = max(target_by_serial) + 1
    output_serial_by_reference: Dict[int, int] = {}
    for reference_atom in reference_document.atoms:
        if serial_policy == "reference":
            output_serial_by_reference[reference_atom.serial] = reference_atom.serial
        elif reference_atom.serial in mapping:
            output_serial_by_reference[reference_atom.serial] = mapping[reference_atom.serial]
        else:
            while next_serial in used_target_serials:
                next_serial += 1
            output_serial_by_reference[reference_atom.serial] = next_serial
            used_target_serials.add(next_serial)
            next_serial += 1

    output_path = Path(output) if output is not None else default_output_path(target)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_lines: List[str] = list(target_document.metadata_lines)
    output_lines.append("REMARK Reindexed using reference atom identity and connectivity")

    target_anisou_by_serial = target_document.anisou_by_serial
    for index, reference_atom in enumerate(reference_document.atoms):
        target_serial = mapping.get(reference_atom.serial)
        output_serial = output_serial_by_reference[reference_atom.serial]
        if target_serial is not None:
            target_atom = target_by_serial[target_serial]
            line = _render_atom(target_atom, reference_atom, output_serial)
            output_lines.append(line)
            if target_serial in target_anisou_by_serial:
                output_lines.append(
                    _render_anisou(
                        target_anisou_by_serial[target_serial], reference_atom, output_serial
                    )
                )
        else:
            point = generated_coordinates.get(reference_atom.serial)
            if point is None:
                raise HydrogenReconciliationError(
                    f"No coordinates available for generated hydrogen {reference_atom.serial}."
                )
            output_lines.append(_render_atom(reference_atom, reference_atom, output_serial, point))
        if target_document.had_ter_records:
            next_atom = reference_document.atoms[index + 1] if index + 1 < len(reference_document.atoms) else None
            if next_atom is None or next_atom.chain_id != reference_atom.chain_id:
                output_lines.append(_render_ter(reference_atom, output_serial + 1))

    output_lines.extend(_conect_lines(reference_mol, output_serial_by_reference))
    output_lines.append("END")
    output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    return ReindexResult(
        output_path=output_path,
        serial_policy=serial_policy,
        reference_to_target=dict(mapping),
        generated_hydrogens=tuple(
            atom.serial for atom in reference_document.atoms if atom.serial not in mapping and atom.element == "H"
        ),
        removed_target_hydrogens=removed_hydrogens,
    )
