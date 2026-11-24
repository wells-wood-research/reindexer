"""Distance-inferred molecular connectivity graphs.

This module deliberately models connectivity only.  It does not perceive bond
orders and does not depend on a cheminformatics toolkit.
"""

from pathlib import Path
from typing import Optional
import logging

import networkx as nx
import numpy as np
import pandas as pd

from .io import pdb2df, xyz2df


#: Single-bond covalent radii, in angstroms, used to decide whether two atoms
#: are bonded.  Every value is Cordero's tabulated radius except phosphorus.
#:
#: **Phosphorus is 0.90 rather than Cordero's 1.07.**  Phosphorus in a
#: biological phosphate is hypervalent, so its bonds are much shorter than a
#: pair of single-bond radii predicts, while its non-bonded neighbours sit
#: closer than in ordinary organic chemistry.  Both effects push the same way,
#: and at 1.07 the two ends collide: in FAD's pyrophosphate the two phosphorus
#: atoms are 2.584 A apart and not bonded to each other, but the cutoff
#: 1.07 + 1.07 + 0.45 = 2.59 A perceives a bond between them.  That one
#: spurious edge changes the connectivity graph, so the atom matching finds no
#: isomorphism and the whole ligand fails to sanitise.
#:
#: 0.90 is chosen to sit clear of both failure modes at the default tolerance
#: of 0.45, measured against the real FAD adduct geometry:
#:
#:   bond or contact        distance   cutoff at 0.90   margin
#:   P-O, longest ester      1.611 A       2.01 A       0.40 A bonded
#:   P-P, diphosphine        2.21  A       2.25 A       0.04 A bonded
#:   P...C, 1-3 contact      2.442 A       2.11 A       0.33 A not bonded
#:   P...P, pyrophosphate    2.584 A       2.25 A       0.33 A not bonded
#:
#: A genuine P-P single bond is still perceived, so the narrower radius buys
#: the pyrophosphate fix without giving up real phosphorus-phosphorus
#: chemistry.  Lowering ``tol_bond`` instead would have worked here too, but
#: that tolerance applies to every element and was chosen to match how
#: molecular viewers draw bonds, so the element-specific problem is fixed with
#: an element-specific number.
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "P": 0.90,
    "S": 1.05,
    "Cl": 1.02,
    "Br": 1.20,
    "I": 1.39,
}


def cov_radius(element: str) -> float:
    """Return the covalent radius used by distance-based edge inference."""
    return COVALENT_RADII.get(str(element).strip(), 0.6)


_PDB_NODE_COLUMNS = {
    "ATOM_ID": "pdb_serial",
    "ATOM_NAME": "atom_name",
    "RES_NAME": "residue_name",
    "RES_ID": "residue_id",
    "CHAIN_ID": "chain_id",
    "OCCUPANCY": "occupancy",
    "BETAFACTOR": "bfactor",
}


def _element(value: object, atom_idx: int) -> str:
    element = str(value).strip()
    if not element:
        raise ValueError(f"Atom {atom_idx} has an empty ELEMENT value")
    return element[0].upper() + element[1:].lower()


def xyz2graph(
    pathXYZ: str | Path,
    tol_bond: float = 0.45,
    logger: Optional[logging.Logger] = None,
):
    """Read an XYZ file and build its distance-inferred connectivity graph."""
    xyz_data = xyz2df(pathXYZ, logger)
    if xyz_data is None:
        raise ValueError(f"Failed to read XYZ file: {pathXYZ}")
    _, _, df = xyz_data
    return df2graph(df, tol_bond=tol_bond, logger=logger)


def pdb2graph(
    pathPDB: str | Path,
    tol_bond: float = 0.45,
    logger: Optional[logging.Logger] = None,
):
    """Read a PDB file and build its distance-inferred connectivity graph."""
    df = pdb2df(pathPDB, logger)
    if df is None:
        raise ValueError(f"Failed to read PDB file: {pathPDB}")
    return df2graph(df, tol_bond=tol_bond, logger=logger)


def df2graph(
    df: pd.DataFrame,
    tol_bond: float = 0.45,
    logger: Optional[logging.Logger] = None,
) -> nx.Graph:
    """Convert atom rows into a NetworkX connectivity graph.

    Nodes represent atoms and retain zero-based internal indices in
    ``original_index``.  Edges represent bonds inferred solely from pairwise
    distances, covalent radii, and ``tol_bond``.  No bond order is assigned.
    """
    if df is None:
        raise ValueError("Input DataFrame is None")
    required = {"ELEMENT", "X", "Y", "Z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    graph = nx.Graph()
    for atom_idx, (_, row) in enumerate(df.iterrows()):
        try:
            coords = np.asarray([row["X"], row["Y"], row["Z"]], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Atom {atom_idx} has invalid coordinates") from exc
        if coords.shape != (3,) or not np.isfinite(coords).all():
            raise ValueError(f"Atom {atom_idx} has invalid coordinates: {coords!r}")

        attrs = {
            "element": _element(row["ELEMENT"], atom_idx),
            "coords": coords.copy(),
            "original_index": atom_idx,
            "source_row": df.index[atom_idx],
        }
        for source_column, node_attribute in _PDB_NODE_COLUMNS.items():
            if source_column in df.columns:
                value = row[source_column]
                attrs[node_attribute] = value.item() if hasattr(value, "item") else value
        graph.add_node(atom_idx, **attrs)

    nodes = list(graph.nodes)
    for position, node1 in enumerate(nodes):
        coords1 = graph.nodes[node1]["coords"]
        for node2 in nodes[position + 1 :]:
            coords2 = graph.nodes[node2]["coords"]
            distance = float(np.linalg.norm(coords1 - coords2))
            cutoff = cov_radius(graph.nodes[node1]["element"]) + cov_radius(
                graph.nodes[node2]["element"]
            ) + tol_bond
            if distance < cutoff:
                graph.add_edge(node1, node2, distance=distance, cutoff=cutoff)
    return graph
