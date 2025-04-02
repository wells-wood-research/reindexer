class XYZFileFormatError(Exception):
    """Exception raised for errors in the XYZ file format."""
    pass

class SubstructureNotFound(Exception):
    """Exception raised when RDKit's findMCS (maximum common substructure) calculation failed to finish."""
    pass