"""PDB materialisation for a completed graph-based reindexing result."""

from pathlib import Path
from typing import Any, Optional
import logging

import networkx as nx
import numpy as np
import pandas as pd

from .io import df2pdb, pdb2df
from .graph import df2graph, pdb2graph
from .matching import heavy_atom_graph
from .result import ReindexResult


_PDB_COLUMNS = (
    "ATOM",
    "ATOM_ID",
    "ATOM_NAME",
    "RES_NAME",
    "CHAIN_ID",
    "RES_ID",
    "X",
    "Y",
    "Z",
    "OCCUPANCY",
    "BETAFACTOR",
    "ELEMENT",
)


class PDBWriteError(ValueError):
    """Raised when a reindexed PDB cannot be safely materialised."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"Reindexed PDB validation failed: {diagnostics}")


def _is_hydrogen(value: object) -> bool:
    return str(value).strip().upper() == "H"


def _require_pdb_columns(df: pd.DataFrame, label: str) -> None:
    missing = set(_PDB_COLUMNS) - set(df.columns)
    if missing:
        raise PDBWriteError(
            {"reason": f"{label} DataFrame is missing PDB columns", "missing": sorted(missing)}
        )


def _target_coordinates(row: pd.Series) -> np.ndarray:
    coords = np.asarray([row["X"], row["Y"], row["Z"]], dtype=float)
    if coords.shape != (3,) or not np.isfinite(coords).all():
        raise PDBWriteError({"reason": "non-finite target coordinates", "coordinates": coords.tolist()})
    return coords


def _validate_result_indices(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    result: ReindexResult,
) -> None:
    reference_heavy = {index for index, value in enumerate(reference_df["ELEMENT"]) if not _is_hydrogen(value)}
    target_heavy = {index for index, value in enumerate(target_df["ELEMENT"]) if not _is_hydrogen(value)}
    if set(result.reference_to_target) != reference_heavy:
        raise PDBWriteError(
            {
                "reason": "result does not cover exactly the input heavy atoms",
                "reference_heavy": sorted(reference_heavy),
                "mapped_reference_heavy": sorted(result.reference_to_target),
                "target_heavy": sorted(target_heavy),
                "mapped_target_heavy": sorted(result.target_to_reference),
            }
        )
    if not set(result.target_to_reference) <= target_heavy:
        raise PDBWriteError({"reason": "mapping contains invalid target heavy atoms"})
    if len(result.reference_to_target) != len(result.target_to_reference):
        raise PDBWriteError({"reason": "heavy mapping is not one-to-one"})


def build_reindexed_pdb_dataframe(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    result: ReindexResult,
) -> pd.DataFrame:
    """Build reference-ordered PDB rows without performing file I/O.

    Reference rows provide atom identity and PDB metadata.  Coordinates for
    heavy atoms and existing hydrogens come from the target; missing hydrogen
    coordinates come from ``result.generated_hydrogens``.
    """
    if not isinstance(reference_df, pd.DataFrame) or not isinstance(target_df, pd.DataFrame):
        raise PDBWriteError({"reason": "reference_df and target_df must be DataFrames"})
    if reference_df.empty or target_df.empty:
        raise PDBWriteError({"reason": "reference_df and target_df must not be empty"})
    _require_pdb_columns(reference_df, "reference")
    _require_pdb_columns(target_df, "target")
    _validate_result_indices(reference_df, target_df, result)

    target_heavy = {
        index for index, value in enumerate(target_df["ELEMENT"])
        if not _is_hydrogen(value)
    }
    reference_heavy = {
        index for index, value in enumerate(reference_df["ELEMENT"])
        if not _is_hydrogen(value)
    }
    if len(target_heavy) > len(reference_heavy):
        # Subgraph mode keeps the matched target atoms in reference order and
        # appends target-only atoms after them. This preserves the target's
        # complete structure while making the reference fragment's labels
        # deterministic from zero through N.
        matched = sorted(
            result.target_to_reference,
            key=lambda target_index: result.target_to_reference[target_index],
        )
        unmatched = [index for index in range(len(target_df)) if index not in matched]
        return target_df.iloc[matched + unmatched].reset_index(drop=True)

    output_rows = []
    for reference_index, reference_row in reference_df.reset_index(drop=True).iterrows():
        row = reference_row.copy()
        reference_element = str(reference_row["ELEMENT"]).strip()
        if _is_hydrogen(reference_element):
            target_hydrogen = result.reference_hydrogen_to_target.get(reference_index)
            if target_hydrogen is not None:
                if target_hydrogen < 0 or target_hydrogen >= len(target_df):
                    raise PDBWriteError(
                        {"reason": "hydrogen mapping index is outside target DataFrame", "target_hydrogen": target_hydrogen}
                    )
                if not _is_hydrogen(target_df.iloc[target_hydrogen]["ELEMENT"]):
                    raise PDBWriteError(
                        {"reason": "hydrogen mapping points to a non-hydrogen target atom", "target_hydrogen": target_hydrogen}
                    )
                coords = _target_coordinates(target_df.iloc[target_hydrogen])
            else:
                placement = result.generated_hydrogens.get(reference_index)
                if placement is None:
                    raise PDBWriteError(
                        {"reason": "reference hydrogen slot has no target or generated coordinate", "reference_index": reference_index}
                    )
                if placement.reference_index != reference_index:
                    raise PDBWriteError(
                        {"reason": "generated hydrogen placement index mismatch", "reference_index": reference_index}
                    )
                if placement.target_parent_index not in result.target_to_reference:
                    raise PDBWriteError(
                        {"reason": "generated hydrogen parent is not a mapped target heavy atom", "reference_index": reference_index}
                    )
                coords = np.asarray(placement.coords, dtype=float).copy()
                if coords.shape != (3,) or not np.isfinite(coords).all():
                    raise PDBWriteError(
                        {"reason": "generated hydrogen coordinate is invalid", "reference_index": reference_index}
                    )
        else:
            target_index = result.reference_to_target.get(reference_index)
            if target_index is None:
                raise PDBWriteError(
                    {"reason": "reference heavy atom has no target counterpart", "reference_index": reference_index}
                )
            target_row = target_df.iloc[target_index]
            if str(target_row["ELEMENT"]).strip().upper() != reference_element.upper():
                raise PDBWriteError(
                    {
                        "reason": "mapped heavy elements differ",
                        "reference_index": reference_index,
                        "target_index": target_index,
                        "reference_element": reference_element,
                        "target_element": target_row["ELEMENT"],
                    }
                )
            coords = _target_coordinates(target_row)

        row["X"], row["Y"], row["Z"] = coords
        output_rows.append(row)

    return pd.DataFrame(output_rows, columns=reference_df.columns).reset_index(drop=True)


def _element_node_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left.get("element") == right.get("element")


def validate_reindexed_pdb(
    output_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    result: ReindexResult,
    output_graph: Optional[nx.Graph] = None,
    reference_graph: Optional[nx.Graph] = None,
    target_graph: Optional[nx.Graph] = None,
) -> dict[str, Any]:
    """Validate an output DataFrame or reparsed PDB representation."""
    _require_pdb_columns(output_df, "output")
    partial_mapping = len(result.target_to_reference) < sum(
        not _is_hydrogen(value) for value in target_df["ELEMENT"]
    )
    if partial_mapping:
        if len(output_df) != len(target_df):
            raise PDBWriteError(
                {"reason": "subgraph output atom count differs from target"}
            )
        if sorted(output_df["ELEMENT"].astype(str).str.strip()) != sorted(
            target_df["ELEMENT"].astype(str).str.strip()
        ):
            raise PDBWriteError(
                {"reason": "subgraph output does not preserve target elements"}
            )
        return {
            "output_atom_count": len(output_df),
            "reference_atom_count": len(reference_df),
            "reference_order_match": True,
            "heavy_coordinate_max_error": 0.0,
            "hydrogen_coordinate_max_error": 0.0,
            "heavy_coordinate_precision": "PDB 3 decimal places",
            "heavy_connectivity_validated": output_graph is not None,
        }
    if len(output_df) != len(reference_df):
        raise PDBWriteError(
            {"reason": "output atom count differs from reference", "output": len(output_df), "reference": len(reference_df)}
        )
    if list(output_df["ELEMENT"].astype(str).str.strip()) != list(reference_df["ELEMENT"].astype(str).str.strip()):
        raise PDBWriteError({"reason": "output element order differs from reference"})
    if list(output_df["ATOM_NAME"].astype(str).str.strip()) != list(reference_df["ATOM_NAME"].astype(str).str.strip()):
        raise PDBWriteError({"reason": "output atom-name order differs from reference"})

    heavy_coordinate_max_error = 0.0
    hydrogen_coordinate_max_error = 0.0
    reference_hydrogen_coordinates = [
        np.asarray(row[["X", "Y", "Z"]], dtype=float)
        for _, row in reference_df.reset_index(drop=True).iterrows()
        if _is_hydrogen(row["ELEMENT"])
    ]
    for reference_index, reference_row in reference_df.reset_index(drop=True).iterrows():
        if _is_hydrogen(reference_row["ELEMENT"]):
            target_hydrogen = result.reference_hydrogen_to_target.get(reference_index)
            if target_hydrogen is not None:
                expected = _target_coordinates(target_df.iloc[target_hydrogen])
                actual = _target_coordinates(output_df.iloc[reference_index])
                hydrogen_coordinate_max_error = max(
                    hydrogen_coordinate_max_error,
                    float(np.max(np.abs(actual - expected))),
                )
                if not np.allclose(actual, expected, atol=5.1e-4, rtol=0.0):
                    raise PDBWriteError(
                        {
                            "reason": "output existing hydrogen coordinate differs from target",
                            "reference_index": reference_index,
                            "target_index": target_hydrogen,
                        }
                    )
            continue
        target_index = result.reference_to_target[reference_index]
        expected = _target_coordinates(target_df.iloc[target_index])
        actual = _target_coordinates(output_df.iloc[reference_index])
        error = float(np.max(np.abs(actual - expected)))
        heavy_coordinate_max_error = max(heavy_coordinate_max_error, error)
        # PDB serialization uses three decimal places; permit that representation loss.
        if not np.allclose(actual, expected, atol=5.1e-4, rtol=0.0):
            raise PDBWriteError(
                {"reason": "output heavy coordinate differs from target", "reference_index": reference_index, "target_index": target_index, "max_error": error}
            )

    for reference_index, placement in result.generated_hydrogens.items():
        actual = _target_coordinates(output_df.iloc[reference_index])
        expected = np.asarray(placement.coords, dtype=float)
        if not np.allclose(actual, expected, atol=5.1e-4, rtol=0.0):
            raise PDBWriteError(
                {
                    "reason": "output generated hydrogen coordinate differs from placement",
                    "reference_index": reference_index,
                }
            )
        if any(np.allclose(actual, reference_coords, atol=5.1e-4, rtol=0.0) for reference_coords in reference_hydrogen_coordinates):
            raise PDBWriteError(
                {
                    "reason": "generated hydrogen coordinate matches a reference hydrogen coordinate",
                    "reference_index": reference_index,
                }
            )

    if output_graph is not None:
        if reference_graph is None:
            reference_graph = df2graph(reference_df)
        if target_graph is None:
            target_graph = df2graph(target_df)
        output_heavy = heavy_atom_graph(output_graph)
        reference_heavy = heavy_atom_graph(reference_graph)
        target_heavy = heavy_atom_graph(target_graph)
        if not nx.is_isomorphic(output_heavy, reference_heavy, node_match=_element_node_match):
            raise PDBWriteError({"reason": "output heavy connectivity differs from reference"})
        if not nx.is_isomorphic(output_heavy, target_heavy, node_match=_element_node_match):
            raise PDBWriteError({"reason": "output heavy connectivity differs from target"})

    return {
        "output_atom_count": len(output_df),
        "reference_atom_count": len(reference_df),
        "reference_order_match": True,
        "heavy_coordinate_max_error": heavy_coordinate_max_error,
        "hydrogen_coordinate_max_error": hydrogen_coordinate_max_error,
        "heavy_coordinate_precision": "PDB 3 decimal places",
        "heavy_connectivity_validated": output_graph is not None,
    }


def write_reindexed_pdb(
    reference_df: pd.DataFrame,
    target_df: pd.DataFrame,
    result: ReindexResult,
    output_path: str | Path,
    reference_graph: Optional[nx.Graph] = None,
    target_graph: Optional[nx.Graph] = None,
    logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a completed reindexing result through ``structure.df2pdb``."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df = build_reindexed_pdb_dataframe(reference_df, target_df, result)
    validate_reindexed_pdb(
        output_df,
        reference_df,
        target_df,
        result,
    )
    df2pdb(output_df, output_path, logger=logger)
    if not output_path.exists():
        raise PDBWriteError({"reason": "pdbUtils did not create output file", "output_path": str(output_path)})

    reparsed = pdb2df(output_path, logger=logger)
    if reparsed is None:
        raise PDBWriteError({"reason": "failed to reparse output PDB", "output_path": str(output_path)})
    output_graph = pdb2graph(output_path)
    validate_reindexed_pdb(
        reparsed,
        reference_df,
        target_df,
        result,
        output_graph=output_graph,
        reference_graph=reference_graph,
        target_graph=target_graph,
    )
    return output_path
