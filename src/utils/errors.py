class XYZFileFormatError(Exception):
    """Exception raised for errors in the XYZ file format."""
    pass

class IsomerMismatchError(Exception):
    """Exception raised when two molecules are not isomers and cannot be matched."""
    pass