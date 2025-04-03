from pdbUtils import pdbUtils
import pandas as pd
import argparse
import os
import re
from typing import Optional, List

from rdkit import Chem as RDchem
from rdkit.Chem import Draw
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import AllChem as RDchem
from rdkit.Chem import rdFMCS

from utils.errors import XYZFileFormatError, SubstructureNotFound

def load_molecule(molec: str) -> tuple[pd.DataFrame, RDchem.Mol, str]:
    """
    Read molecule structure file into a pandas dataframe, handing PDB and XYZ file formats.

    Args:
        molec (str): Absolute path to a molecule structure file.

    Returns:
        pd.DataFrame:   Dataframe describing the input structure with columns:
                        ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']
        RDchem.Mol:     RDKit molecule object constructred from the input structure file
        str:            Canonical SMILES string for a molecule
    """
    name, extension = get_file_format(molec)
    if extension == "pdb":
        df = pdbUtils.pdb2df(molec)
        mol = RDchem.rdmolfiles.MolFromPDBFile(molec)
    else:
        # Assume .xyz as checked at get_file_format level
        df = xyz2df(molec)
        mol = RDchem.rdmolfiles.MolFromXYZFile(molec)
        RDchem.rdDetermineBonds.DetermineBonds(mol,charge=0) # TODO Must allow user to input their charge
    mol.SetProp("name", name)
    smiles = RDchem.rdmolfiles.MolToSmiles(mol, canonical=True)
    return df, mol, smiles

def get_file_format(file: str) -> Optional[tuple[str, str]]:
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
    """
    # Type checking
    if not isinstance(file, str):
        raise TypeError("Input must be a string representing a file path.")

    # Check if file path is empty
    if not file.strip():
        raise ValueError("File path cannot be empty.")

    if not is_valid_path(file):
        raise ValueError("File path cannot be empty.")
    try:
        # Get the extension (with the dot)
        root, extension = os.path.splitext(file)
        name = os.path.basename(root)
        # Check if there is no extension
        if not extension:
            raise ValueError("Please provide a full file path with an extension (e.g., filename.pdb or filename.xyz).")

        # Define supported extensions (as lowercase for consistency)
        global IMPLEMENTED_EXTENSIONS
        IMPLEMENTED_EXTENSIONS = {".pdb", ".xyz"}  # Using a set for faster lookup
        if extension not in IMPLEMENTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension '{extension}'. Allowed extensions: {sorted(IMPLEMENTED_EXTENSIONS)}")

        return name, extension.lstrip('.')

    except AttributeError:
        raise TypeError("File path must be a string or path-like object.")
    except Exception as e:
        raise RuntimeError(f"Error processing file: {str(e)}") from e

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

def xyz2df(molec: str) -> pd.DataFrame:
    """
    CREDIT: Inspired by pdbUtils (https://github.com/ESPhoenix/pdbUtils) code by Eugene Shrimpton-Pheonix.

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
                noAtms = int(lines[0].strip())
            except ValueError:
                raise XYZFileFormatError("First line of XYZ file must be an integer (number of atoms).")

            # Verify that the number of remaining lines matches the number of atoms
            atom_lines = lines[2:]  # Skip the first two header lines
            if len(atom_lines) != noAtms:
                raise XYZFileFormatError(f"Expected {noAtms} atoms, but found {len(atom_lines)} data lines.")

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

def df2xyz(df: pd.DataFrame, filepath: str) -> None:
    """
    Save a pandas DataFrame containing atomic structure data into an XYZ file.

    The first line contains the number of atoms (rows in the DataFrame), the second line is a comment line, and subsequent lines 
    list the atom label (from the 'ELEMENT' column of the dataframe) and X, Y, Z coordinates, separated by tabs.

    Args:
        df (pd.DataFrame):  DataFrame describing the atomic structure with columns:
                            ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 
                            'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']. Must include at least 'ELEMENT', 'X', 'Y', 'Z'.
        filepath (str):     Absolute path where the XYZ file will be saved, including the '.xyz' extension.

    Raises:
        KeyError: If required columns ('ELEMENT', 'X', 'Y', 'Z') are missing from the DataFrame.
        IOError: If there is an issue writing to the specified file path.
    """
    # Check for required columns
    colRequired = ['ELEMENT', 'X', 'Y', 'Z']
    if not all(col in df.columns for col in colRequired):
        raise KeyError(f"DataFrame is missing required columns: {colRequired}")

    # Number of atoms is the number of rows in the DataFrame
    noAtms = len(df)

    # Prepare the header lines
    header = f"{noAtms}\nSaved from a dataframe by https://github.com/wells-wood-research/reindexer.git\n"

    # Format each row as "ELEMENT    X    Y    Z" with tab separation and fixed-width floating-point notation
    atom_lines = []
    for _, row in df.iterrows():
        element = f"{row['ELEMENT']:<2}"  # Left-align element, fixed width of 2 (e.g., 'C ', 'N ')
        x = f"{row['X']:>9.5f}"          # Right-align X, width of 9, 5 decimal places
        y = f"{row['Y']:>9.5f}"          # Right-align Y, width of 9, 5 decimal places
        z = f"{row['Z']:>9.5f}"          # Right-align Z, width of 9, 5 decimal places
        line = f"{element}\t{x}\t{y}\t{z}"
        atom_lines.append(line)

    # Combine header and atom lines
    xyz_content = header + "\n".join(atom_lines) + "\n"  # Add final newline for consistency

    # Write to file
    try:
        with open(filepath, 'w') as xyz_file:
            xyz_file.write(xyz_content)
    except IOError as e:
        raise IOError(f"Error writing to XYZ file at {filepath}: {str(e)}")

#############################################################################################################

def get_maximum_common_substructure(mol1: RDchem.Mol, mol2: RDchem.Mol) -> RDchem.Mol:
    """
    CREDIT: from RDKit cookbook (https://www.rdkit.org/docs/Cookbook.html#highlight-molecule-differences) code by Takayuki Serizawa.
    
    Finds maximum common substructure between two molecules:
    the largest substructure common to both input molecules.
    FindMCS returns MCSResult object, including the following properties:
        canceled:   if True, the MCS calculation did not finish;    ==> abort
        queryMol:   query molecule for the MCS;                     ==> return object

    Args:
        mol1 (RDchem.Mol):      The reference RDKit molecule object.
        mol2 (RDchem.Mol):      The referee RDKit molecule object to compare against mol1.

    Returns:
        RDchem.Mol:             The maximum common substructure as an RDKit molecule object.

    Raises:
        SubstructureNotFound: If MCS cannot be found by RDKit.
    """
    mcs = rdFMCS.FindMCS([mol1,mol2])
    if mcs.canceled:
        raise SubstructureNotFound("Common substructure could not be found. Investigate your inputs.")
    mcs_mol = mcs.queryMol
    return mcs_mol

#############################################################################################################

def save_image_difference(mol1: RDchem.Mol, match1: tuple[int], mol2: RDchem.Mol, match2: tuple[int], outpath: str) -> None:
    """
    Save an SVG image comparing two molecules, visually highlighting their differences.

    This function generates and saves an SVG image showing the reference and referee 
    molecules side by side, with atom indices labeled and differences highlighted in red. 
    The image is resized to fit the content without excessive margins.

    Args:
        mol1 (RDchem.Mol):      The reference molecule object (expected to be RDchem.Mol).
        match1 (tuple[int]):    The atom indices in mol1 that match the MCS.
        mol2 (RDchem.Mol):      The referee molecule object (expected to be RDchem.Mol).
        match2 (tuple[int]):    The atom indices in mol2 that match the MCS.
        mcs_mol (RDchem.Mol):   The maximum common substructure molecule (RDchem.Mol).
        outpath (str):          The output path (without extension) where the SVG file will be saved.

    Returns:
        None
    """
    label_atom_indices(mol1)
    label_atom_indices(mol2)
    diff1, diff2 = get_atom_difference(mol1, match1), get_atom_difference(mol2, match2)

    mols = [mol1, mol2]
    sub_width = 500
    svg_width = len(mols)*sub_width
    sub_height = 500
    svg_height = sub_height

    img = RDchem.Draw.MolsToGridImage(mols = mols
                                      , subImgSize = (sub_width, sub_height)
                                      , legends = [f"reference: {mol1.GetProp('name')}", f"referee: {mol2.GetProp('name')}"]
                                      , highlightAtomLists=[diff1, diff2]
                                      , useSVG = True
                                      )
    resized_img = edit_image_dimensions(img, svg_width, svg_height)

    with open(f"{outpath}.svg", 'w') as f:
        f.write(resized_img)

def edit_image_dimensions(img: str, width: float, height: float) -> str:
    """
    Resize an SVG image to fit its content by adjusting width, height, and viewBox.

    This function modifies the SVG string to ensure the image dimensions match the 
    content, removing unnecessary margins. It updates the 'width', 'height', and 
    'viewBox' attributes, as well as the background rectangle.

    Args:
        img (str):      input SVG image as a string.
        width (float):  width of the subimages
        height (float): height of the subimages

    Returns:
        str:            The modified SVG string with updated dimensions and viewBox to fit the content.

    Note:
        Small padding (20 pixels) is added to the width and height to ensure no content is cut off.
    """
    # Update SVG string
    resized_img = re.sub(r'width=\'[^\']+\' height=\'[^\']+\' viewBox=\'[^\']+\'', 
                    f'width=\'{width+20}px\' height=\'{height+20}px\' viewBox=\'0.0 0.0 {width} {height}\'', 
                    img, 1)

    # Update the svg rect
    resized_img = re.sub(r'<rect style=[^\>]+ width=\'[^\']+\' height=\'[^\']+\' x=\'[^\']+\' y=\'[^\']+\'>',
                    f'<rect style="opacity:1.0;fill:#FFFFFF;stroke:none" width="{width}" height="{height}" x="0" y="0">',
                    resized_img, 1)
    return resized_img

def label_atom_indices(mol: RDchem.Mol) -> None:
    """
    Label each atom in a molecule with its index.

    This function iterates over all atoms in the input molecule and assigns each 
    atom its index as a map number, which can be used for visualization or comparison.

    Args:
        mol (RDchem.Mol):    The molecule object to be labelled.

    Returns:
        None
    """
    for idx, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(idx)

def get_atom_difference(mol: RDchem.Mol, match: tuple[int]) -> list:
    """
    Identify atoms in two molecules, which differ from their maximum common substructure.

    This function compares the atoms in two molecules against their matches in the 
    maximum common substructure (MCS) to find atoms that do not match. These atoms 
    are returned as lists of indices, which can be used to highlight differences 
    (e.g., in red) in visualizations.

    Args:
        mol (RDchem.Mol):      molecule object
        match (tuple[int]):    list of atom indices in mol that match the MCS
    Returns:
       (list):  List of atom indices in a mol that do not match MCS of mol and another mol.
    """
    diff = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in match:
            diff.append(atom.GetIdx())
    return diff

#############################################################################################################

def split_df_by_H(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame describing molecular structure into two DataFrames: one for non-hydrogen atoms and one for hydrogen atoms.

    The split is determined by checking the 'ELEMENT' column, hydrogen atoms ahave 'H' in 'ELEMENT'.

    Args:
        df (pd.DataFrame):  Dataframe describing the input structure with columns:
                            ['ATOM', 'ATOM_ID', 'ATOM_NAME', 'RES_NAME', 'CHAIN_ID', 'RES_ID', 'X', 'Y', 'Z', 'OCCUPANCY', 'BETAFACTOR', 'ELEMENT']

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames:
            - First DataFrame: Rows where the atom is not hydrogen ('ELEMENT' is not 'H').
            - Second DataFrame: Rows where the atom is hydrogen ('ELEMENT' is 'H').

    Warning:
        If no rows contain hydrogen atoms (i.e., no rows where 'ATOM_NAME' or 'ELEMENT' is 'H'), a warning is printed 
        indicating that no hydrogen atoms were found in the DataFrame. The second DataFrame in this case will be empty.
    """
    # Check if either ATOM_NAME or ELEMENT is 'H' to identify hydrogen atoms
    isH = df['ELEMENT'].str.upper().eq('H')

    # Split DataFrame
    dfNotH = df[~isH].copy()  # Rows where atom is not hydrogen
    dfH = df[isH].copy()       # Rows where atom is hydrogen

    # Warn if no hydrogen atoms are found
    if len(dfH) == 0:
        print("Warning: No hydrogen atoms found in the DataFrame. Have you provided correct input structure, including hydrogens?")

    return dfNotH, dfH

def reindex(df1: pd.DataFrame, df2: pd.DataFrame, match1: tuple[int], match2: tuple[int], diff1: list, diff2: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reindex two DataFrames to match indices between elements that are in common, and appending the unique elements at the end.
    Indices of the reference (df1) are maintained, while referee (df2) is reindexed.

    This function takes a reference DataFrame and a referee DataFrame, along with tuples of indices ('match1' and 
    'match2') that indicate which indices correspond to common elements between the two. It reindexes the referee 
    DataFrame to match the reference DataFrame's indices for common elements, keeps the reference DataFrame's indices 
    unchanged, and appends any additional (uncommon) elements from both DataFrames at the end with new sequential indices.

    Args:
        df1 (pd.DataFrame):     The reference DataFrame whose indices should remain unchanged
        df2 (pd.DataFrame):     The referee DataFrame to be reindexed to match the reference
        match1 (tuple[int]):    Tuple of indices in df1 that correspond to common elements.
        match2 (tuple[int]):    Tuple of indices in df2 that correspond to the same common elements as match1.
        diff1 (list[int]):      List of indices of atoms unique to mol1 (do not match MCS of mol1 and mol2).
        diff2 (list[int]):      List of indices of atoms unique to mol2 (do not match MCS of mol1 and mol2).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - First DataFrame: The reference DataFrame with unchanged indices for common elements and its unique elements 
                              appended at the end.
            - Second DataFrame: The reindexed referee DataFrame where common elements match the reference indices, and 
                               unique elements are appended at the end.

    Note:
        The function assumes that 'match1' and 'match2' are of the same length and correctly map common elements 
        between the two DataFrames. If there are mismatches or inconsistencies, the results may be incorrect.
    """
    # Convert tuples to lists for easier manipulation
    idx1common = list(match1)
    idx2common = list(match2)
    # Get all original indices from both DataFrames
    idx1all = list(df1.index)
    idx2all = list(df2.index)

    # Create new DataFrames for output
    df1new = df1.copy()
    df2new = df2.copy()

    # Reindex referee DataFrame for common elements
    df1common = df1new.loc[idx1common].copy()
    df2common = df2new.loc[idx2common].copy()

    # Assign new indices to referee to match reference
    df2common.index = idx1common

    # Handle additional elements
    # For reference, keep original indices but append at end
    if diff1:
        df1diff = df1new.loc[diff1].copy()
        # Assign new sequential indices starting after the last common index
        idxMaxCommon = max(idx1common) if idx1common else -1
        idx1diff = range(idxMaxCommon + 1, idxMaxCommon + 1 + len(diff1))
        df1diff.index = idx1diff

        # Combine common and additional for reference
        df1new = pd.concat([df1common, df1diff])
    else:
        df1new = df1common
    # For referee, append additional elements with new sequential indices
    if diff2:
        df2diff = df2new.loc[diff2].copy()
        # Assign new sequential indices starting after the last common index
        idxMaxCommon = max(idx1common) if idx1common else -1
        idx2diff = range(idxMaxCommon + 1 + len(diff1) if diff1 else idxMaxCommon + 1,
                                      idxMaxCommon + 1 + len(diff1) + len(diff2))
        df2diff.index = idx2diff

        # Combine common and additional for referee
        df2new = pd.concat([df2common, df2diff])
    else:
        df2new = df2common

    # Ensure reference and referee have the same number of rows if no additional referee elements
    #df1new, df2new = pad_to_match(df1new, df2new)

    return df1new, df2new

def pad_to_match(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ensure both DataFrames have the same number of rows by padding the shorter one with empty rows.

    Args:
        df1 (pd.DataFrame): First DataFrame (e.g., reference).
        df2 (pd.DataFrame): Second DataFrame (e.g., referee).

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple of the two DataFrames, padded to have the same length.
    """
    if len(df1) != len(df2):
        dfLonger = df1 if len(df1) > len(df2) else df2
        dfShorter = df2 if len(df1) > len(df2) else df1
        noRowsMissing = abs(len(df1) - len(df2))
        idxPad = range(dfShorter.index[-1] + 1, dfShorter.index[-1] + 1 + noRowsMissing)
        dfPad = pd.DataFrame(index=idxPad, columns=dfShorter.columns)
        if len(df1) > len(df2):
            df2pda = pd.concat([dfShorter, dfPad])
            return df1, df2pda
        else:
            df1pad = pd.concat([dfShorter, dfPad])
            return df1pad, df2
    return df1, df2

#############################################################################################################

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
    """
    df1, mol1, smiles1 = load_molecule(reference)
    df2, mol2, smiles2 = load_molecule(referee)

    mcs_mol = get_maximum_common_substructure(mol1, mol2)
    match1, match2 = mol1.GetSubstructMatch(mcs_mol), mol2.GetSubstructMatch(mcs_mol)
    diff1, diff2 = get_atom_difference(mol1, match1), get_atom_difference(mol2, match2)
    save_image_difference(mol1, match1, mol2, match2, "/home/mchrnwsk/reindexer/src/tests/image_preindexed")

    df1new, df2new = reindex(df1, df2, match1, match2, diff1, diff2)

    reference_reindexed = "/home/mchrnwsk/reindexer/src/tests/quindoline_reindex.xyz"
    referee_reindexed = "/home/mchrnwsk/reindexer/src/tests/3_anilinoquinoline_reindex.xyz"

    df2xyz(df1new, reference_reindexed)
    df2xyz(df2new, referee_reindexed)

    df1reidx, mol1reidx, smiles1reidx = load_molecule(reference_reindexed)
    df2reidx, mol2reidx, smiles2reidx = load_molecule(referee_reindexed)

    mcs_molreidx = get_maximum_common_substructure(mol1reidx, mol2reidx)
    match1reidx, match2reidx = mol1reidx.GetSubstructMatch(mcs_molreidx), mol2reidx.GetSubstructMatch(mcs_molreidx)
    save_image_difference(mol1reidx, match1reidx, mol2reidx, match2reidx, "/home/mchrnwsk/reindexer/src/tests/image_reindexed")

    # TODO calculate charge and multiplicty with RDkit? https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html#rdkit.Chem.rdmolops.GetFormalCharge
    return '/path/to/molecule2_reindx.pdb'

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