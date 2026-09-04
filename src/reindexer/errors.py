"""Errors raised by the PDB reindexing API."""


class PDBError(ValueError):
    """Base class for invalid or unsupported PDB input."""


class PDBFormatError(PDBError):
    """A PDB record cannot be parsed safely."""


class UnsupportedPDBFeatureError(PDBError):
    """The input uses a PDB feature not supported by the round-trip writer."""


class ChemicalMappingError(PDBError):
    """The reference and target structures cannot be reconciled chemically."""


class AmbiguousAtomMappingError(ChemicalMappingError):
    """More than one chemically valid atom mapping remains."""


class HydrogenReconciliationError(ChemicalMappingError):
    """Reference hydrogenation cannot be applied to the target coordinates."""
