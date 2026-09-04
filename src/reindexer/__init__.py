"""Public API for the reindexer package."""

from .errors import (
    AmbiguousAtomMappingError,
    ChemicalMappingError,
    HydrogenReconciliationError,
    PDBError,
    PDBFormatError,
    StructureNotOptimised,
    SubstructureNotFound,
    UnsupportedPDBFeatureError,
    XYZFileFormatError,
)
from .pdb import (
    PDBAtom,
    PDBDocument,
    ReindexResult,
    default_output_path,
    parse_pdb,
    reindex_pdb,
)

__all__ = [
    "AmbiguousAtomMappingError",
    "ChemicalMappingError",
    "HydrogenReconciliationError",
    "PDBAtom",
    "PDBDocument",
    "PDBError",
    "PDBFormatError",
    "ReindexResult",
    "StructureNotOptimised",
    "SubstructureNotFound",
    "UnsupportedPDBFeatureError",
    "XYZFileFormatError",
    "default_output_path",
    "parse_pdb",
    "reindex_pdb",
]
