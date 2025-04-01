from pdbUtils import pdbUtils
from rdkit import Chem as RDchem
from rdkit.Chem import AllChem as RDchem
import pandas as pd

import argparse
import os
from typing import Optional

def is_same_smiles(molec1, molec2) -> bool:
    '''
    TODO
    Check if SMILES of molecules are the same.
    If True, reindexing can be significantly simplified with RDKit.
    '''
    return False
    
def is_isomer(molec1, molec2) -> bool:
    '''
    TODO
    Check if molecules have the same number and types of atoms
    '''
    return False

def is_valid_path(file: str) -> bool:
    '''
    Check if file exists at the provided path.

    Arguments:
        file: absolute path to molecule structure file
    Returns:
        (bool): True if file exists, False otherwise
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
        IMPLEMENTED_EXTENSIONS = {".pdb", ".xyz"}  # Using a set for faster lookup
        if extension not in IMPLEMENTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{extension}'. Allowed extensions: {sorted(IMPLEMENTED_EXTENSIONS)}")

        return extension.lstrip('.')

    except AttributeError:
        raise TypeError("File path must be a string or path-like object.")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}") from e

def main(molec1, molec2, suffix):
    """
    TODO
    Loads reference molecule (index maintained) and referee molecule (reindexed to match reference).
    Check if molecules are isomers of each other (otherwise raise RuntimeError), and if they have same SMILES strings.
    If yes, then reindex using RDKit; otherwise perform manual reindexing. (TODO compare fragments?)

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
    
    inputLigandDf = pdbUtils.pdb2df(molec1)
    inputLigandMol = RDchem.MolFromPDBFile(molec1, removeHs=True)
    inputLigandSmiles = RDchem.MolToSmiles(inputLigandMol, canonical=True)

    outputLigandDf = pdbUtils.pdb2df(molec2)
    outputLigandMol = RDchem.MolFromPDBFile(molec2, removeHs=True)
    outputLigandSmiles = RDchem.MolToSmiles(outputLigandMol, canonical=True)

    # Check if SMILES agree
    if inputLigandSmiles != outputLigandSmiles:
        raise ValueError(f"""[!ERROR!] Input and output SMILES are not the same.
        input: {inputLigandSmiles}
        output: {outputLigandSmiles}""")
    
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
    args = parser.parse_args()
    molec1 = args.reference
    molec2 = args.referee
    suffix = args.suffix

    try:
        main(molec1, molec2, suffix)
    except (ValueError, TypeError, RuntimeError) as e:
        print(f"Error: {e}")