from pdbUtils import pdbUtils
from rdkit import Chem as RDchem
from rdkit.Chem import AllChem as RDchem
import pandas as pd

import argparse
import os
from typing import Optional, List

from utils.errors import XYZFileFormatError, IsomerMismatchError

def is_same_smiles(molec1, molec2) -> bool:
    '''
    TODO
    Check if SMILES of molecules are the same.
    If True, reindexing can be significantly simplified with RDKit.
    '''
    return False
    
def is_isomer(molec1: pd.DataFrame, molec2: pd.DataFrame) -> bool:
    '''
    Check if molecules have the same number and types of atoms.

    Args:
        molec1 (pd.DataFrame): Reference molecule.
        molec2 (pd.DataFrame): Referee molecule.

    Returns:
        bool: True if the molecules are isomers (same number and types of atoms), False otherwise.
    '''
    try:
        # Check if number of lines (rows) matches
        if len(molec1) != len(molec2):
            return False

        # Check if number of occurrences of entries in ELEMENT is the same
        # Get the element counts for both DataFrames
        elements1 = molec1['ELEMENT'].value_counts().sort_index()
        elements2 = molec2['ELEMENT'].value_counts().sort_index()

        # Compare the series. They must be identical (same indices and values)
        return elements1.equals(elements2)

    except Exception as e:
        raise ValueError(f"Error comparing molecules: {str(e)}") from e

def is_valid_path(file: str) -> bool:
    '''
    Check if file exists at the provided path.

    Arguments:
        file: absolute path to molecule structure file
    Returns:
        bool: True if file exists, False otherwise
    '''
    exists = os.path.isfile(file)
    return exists

def get_file_format(file: str) -> Optional[str]:
    """
    Check file format by matching extension string against implemented options.

    Args:
        file (str): Absolute path to a molecule structure file.

    Returns:
        Optional[str]: Extension format of the input file without the leading dot,
                       or None if the format is not supported.

    Raises:
        TypeError: If the input is not a string.
        ValueError: If the file path is empty or has no extension.

    Example:
        >>> get_file_format("/path/to/molecule.pdb")
        'pdb'
    """
    # Type checking
    if not isinstance(file, str):
        raise TypeError("Input must be a string representing a file path.")

    # Check if file path is empty
    if not file.strip():
        raise ValueError("File path cannot be empty.")

    try:
        # Get the extension (with the dot)
        _, extension = os.path.splitext(file)
        
        # Check if there is no extension
        if not extension:
            raise ValueError("Please provide a full file path with an extension (e.g., filename.pdb or filename.xyz).")

        # Define supported extensions (as lowercase for consistency)
        global IMPLEMENTED_EXTENSIONS
        IMPLEMENTED_EXTENSIONS = {".pdb", ".xyz"}  # Using a set for faster lookup
        if extension not in IMPLEMENTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{extension}'. Allowed extensions: {sorted(IMPLEMENTED_EXTENSIONS)}")

        return extension.lstrip('.')

    except AttributeError:
        raise TypeError("File path must be a string or path-like object.")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}") from e

def xyz2df(molec: str) -> pd.DataFrame:
    """
    Read .xyz file into a pandas dataframe.

    Args:
        molec (str): Absolute path to a molecule structure file.

    Returns:
        pd.DataFrame:   Dataframe describing the input structure with columns:
                        ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']

    Raises:
        FileNotFoundError: If the XYZ file does not exist.
        XYZFileFormatError: If the XYZ file format is invalid (e.g., missing header lines).
        RuntimeError: If there are issues reading the file or parsing data.

    Example:
        >>> xyz2df("/home/mchrnwsk/reindexer/src/tests/water.xyz")
            ATOM  ATOM_ID ATOM_NAME RES_NAME CHAIN_ID  RES_ID      X       Y    Z  OCCUPANCY  BETAFACTOR ELEMENT
        0  HETATM        1         O      UNK        A       1  0.000 -0.0589  0.0        1.0         0.0       O
        1  HETATM        2         H      UNK        A       2 -0.811  0.4677  0.0        1.0         0.0       H
        2  HETATM        3         H      UNK        A       3  0.811  0.4677  0.0        1.0         0.0       H
    """
    columns = ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']
    data: List[List] = []

    try:
        with open(molec, 'r') as xyz_file:
            lines = xyz_file.readlines()

            # Check that first two lines are for number of atoms and comment
            if len(lines) < 2:
                raise XYZFileFormatError("Please provide .xyz files with header lines: total number of atoms in line 1, and description in line 2.")

            # First line should be the number of atoms
            try:
                num_atoms = int(lines[0].strip())
            except ValueError:
                raise XYZFileFormatError("First line of XYZ file must be an integer (number of atoms).")

            # Verify that the number of remaining lines matches the number of atoms
            atom_lines = lines[2:]  # Skip the first two header lines
            if len(atom_lines) != num_atoms:
                raise XYZFileFormatError(f"Expected {num_atoms} atoms, but found {len(atom_lines)} data lines.")

            # Process atom data starting from line 3
            for line_num, line in enumerate(atom_lines, start=3):  # Start numbering from 3
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                # Split the line into columns (assuming space-separated)
                parts = line.split()
                if len(parts) < 4:  # Ensure at least atom name, x, y, z are present
                    raise XYZFileFormatError(f"Line {line_num} in XYZ file has insufficient columns. Expected at least 4 (atom, x, y, z).")

                atom_name = parts[0].strip()  # First column: atom name or element
                try:
                    x = float(parts[1].strip())  # Second column: x coordinate
                    y = float(parts[2].strip())  # Third column: y coordinate
                    z = float(parts[3].strip())  # Fourth column: z coordinate
                except (ValueError, IndexError) as e:
                    raise XYZFileFormatError(f"Invalid coordinate data on line {line_num}: {str(e)}")

                # Fill other columns as specified
                atom_type = "HETATM"
                res_name = "UNK"
                chain_id = "A"
                atom_id = line_num - 2  # Line number minus header lines (1-based for users)
                res_id = line_num - 2   # Same as atom_id for simplicity
                occupancy = 1.00
                temp_factor = 0.00
                element = atom_name  # Element is same as atom name

                data.append([atom_type, atom_id, atom_name, res_name, chain_id, res_id, x, y, z, occupancy, temp_factor, element])

        if not data:
            raise XYZFileFormatError("No valid atom data found in XYZ file.")

        return pd.DataFrame(data, columns=columns)

    except FileNotFoundError:
        raise FileNotFoundError(f"XYZ file not found: {molec}")
    except Exception as e:
        raise RuntimeError(f"Error reading XYZ file: {str(e)}") from e

def load_molecule(molec):
    """
    Read molecule structure file into a pandas dataframe, handing PDB and XYZ file formats.

    Args:
        molec (str): Absolute path to a molecule structure file.

    Returns:
        pd.DataFrame:   Dataframe describing the input structure with columns:
                        ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']
    """
    extension = get_file_format(molec)
    if extension == "pdb":
        df = pdbUtils.pdb2df(molec)
    else:
        # Assume .xyz as checked at get_file_format level
        df = xyz2df(molec)
    return df

def main(reference, referee, suffix, outFormat):
    """
    TODO
    Loads reference molecule (index maintained) and referee molecule (reindexed to match reference).
    Check if molecules are isomers of each other (otherwise raise RuntimeError), and if they have same SMILES strings.
    If yes, then reindex using RDKit; otherwise perform manual reindexing. (TODO compare fragments?)

    Args:
        reference (str):       Absolute path to a reference molecule structure file.
        referee (str):       Absolute path to a referee molecule structure file.
        suffix (str):       Suffix to append to name of referee molecule when saving with reindexed atoms.
        outFormat (str):    Structure file format to save output in, choices allowed from IMPLEMENTED_EXTENSIONS

    Returns:
        TODO
        Optional[str]: Extension format of the input file without the leading dot,
                       or None if the format is not supported.

    Raises:
        TODO
        TypeError: If the input is not a string.
        ValueError: If the file path is empty or has no extension.

    Example:
        >>> main("/path/to/molecule.xyz", "/path/to/molecule2.xyz", "_reindx", "pdb")
        '/path/to/molecule2_reindx.pdb'
    """
    molec1 = load_molecule(reference)
    molec2 = load_molecule(referee)

    if not is_isomer(molec1, molec2):
        raise IsomerMismatchError(f"Reference and referee molecules are not isomers (do not contain same number and type of atoms) and cannot be matched.")

###########################################################################
    molec1Mol = RDchem.MolFromPDBFile(molec1, removeHs=True)
    molec1Smiles = RDchem.MolToSmiles(molec1Mol, canonical=True)
    molec2Mol = RDchem.MolFromPDBFile(molec2, removeHs=True)
    molec2Smiles = RDchem.MolToSmiles(molec2Mol, canonical=True)

    # Check if SMILES agree
    if molec1Smiles != molec2Smiles:
        raise ValueError(f"""[!ERROR!] Input and output SMILES are not the same.
        input: {molec1Smiles}
        output: {molec2Smiles}""")
    
    # Returns the indices of the molecule’s atoms that match a substructure query.
    matches = outputLigandMol.GetSubstructMatch(inputLigandMol)
    
    # Ensure the length of the match corresponds to the number of atoms
    if len(matches) != inputLigandMol.GetNumAtoms():
        raise ValueError("Atoms do not match between input and output.")
    
    indexMap = {inputIdx: outputIdx for inputIdx, outputIdx in enumerate(matches)}

    # Reorder carbon atoms coordinates
    reorderedLigandDf = inputLigandDf.copy(deep=True)
    for inputIdx, outputIdx in indexMap.items():
        reorderedLigandDf.loc[inputIdx, ["X", "Y", "Z"]] = outputLigandDf.loc[outputIdx, ["X", "Y", "Z"]].values

    # Drop hydrogen rows from reorderedLigandDf and add from outputLigandDf
    reorderedLigandDf = reorderedLigandDf[~reorderedLigandDf["ATOM_NAME"].str.startswith('H')]
    hydrogenRows = outputLigandDf[outputLigandDf["ATOM_NAME"].str.startswith('H')]
    reorderedLigandDf = pd.concat([reorderedLigandDf, hydrogenRows], ignore_index=True)

    # Save the reordered ligand
    outputFilename = molec2.split(".")[:-1][0]
    reorderedLigandOutpath = f"{outputFilename}_reord.pdb"
    pdbUtils.df2pdb(reorderedLigandDf, reorderedLigandOutpath)
    return reorderedLigandOutpath

# Entry point for command-line execution
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Generate a reindexed structure file of the referee to match atom indices and labels to reference input")
    parser.add_argument("--reference", type=str, help="Path to structure file of a molecule whose atom indices are matched")
    parser.add_argument("--referee", type=str, help="Path to structure file of a molecule to be reordered")
    parser.add_argument("--suffix", type=str, default="_reord", help="Suffix to append to output with reindexed referee structure file")
    parser.add_argument("--outFormat", type=str, choices = IMPLEMENTED_EXTENSIONS, default="xyz", help="Structure file format to save output in")
    args = parser.parse_args()
    reference = args.reference
    referee = args.referee
    suffix = args.suffix
    outFormat = args.outFormat

    try:
        main(reference, referee, suffix, outFormat)
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error: {e}")