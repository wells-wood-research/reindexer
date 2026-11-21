"""Graph-based molecular protonation and reindexing to match reference molecule.

Reads PDB, XYZ, and component-definition mmCIF (``_chem_comp_atom``, see
:func:`reindexer.io.cif2df`).

The reindexing is based on the graph isomorphism between the reference molecule and the target molecule, which is used to relabel the atoms in the target molecule to match the reference molecule.

The reindexing can be performed with or without hydrogen optimisation. The hydrogen optimisation is performed using ORCA (optH - all heavy atoms frozen).
"""

from .pipeline import reindex, reindex_graphs
from .result import HydrogenPlacement, ReindexResult

__version__ = "0.1.0"

__all__ = [
    "HydrogenPlacement",
    "ReindexResult",
    "reindex",
    "reindex_graphs",
]
