"""Compatibility imports for the pre-package exception path.

New code should import exceptions from :mod:`reindexer.errors`.
"""

from reindexer.errors import (
    StructureNotOptimised,
    SubstructureNotFound,
    XYZFileFormatError,
)

__all__ = [
    "StructureNotOptimised",
    "SubstructureNotFound",
    "XYZFileFormatError",
]
