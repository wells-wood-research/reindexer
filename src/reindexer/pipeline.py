"""Composition of the first, non-writing reindexing slice."""

from collections import Counter
from pathlib import Path
from typing import Optional
import logging

import networkx as nx

from .io import pdb2df
from .graph import pdb2graph
from .hydrogens import analyse_hydrogens, generate_missing_hydrogens
from .matching import heavy_atom_graph, match_heavy_atoms
from .optimization import optimise_reindexed_pdb
from .result import ReindexResult
from .writer import write_reindexed_pdb


def reindex_graphs(reference: nx.Graph, target: nx.Graph) -> ReindexResult:
    """Match two molecular graphs and analyse target hydrogen completeness."""
    reference_heavy = heavy_atom_graph(reference)
    target_heavy = heavy_atom_graph(target)
    outcome = match_heavy_atoms(reference_heavy, target_heavy)
    target_to_reference = dict(outcome.mapping)
    reference_to_target = {reference_node: target_node for target_node, reference_node in target_to_reference.items()}
    result = ReindexResult(
        target_to_reference=target_to_reference,
        reference_to_target=reference_to_target,
        mapping_scores=outcome.scores,
        validation={
            "heavy_atom_count_reference": reference_heavy.number_of_nodes(),
            "heavy_atom_count_target": target_heavy.number_of_nodes(),
            "heavy_edge_count_reference": reference_heavy.number_of_edges(),
            "heavy_edge_count_target": target_heavy.number_of_edges(),
            "heavy_isomorphic": True,
            "one_to_one_heavy_mapping": len(target_to_reference)
            == len(reference_to_target)
            == reference_heavy.number_of_nodes(),
            "reference_element_histogram": dict(
                sorted(
                    Counter(nx.get_node_attributes(reference_heavy, "element").values()).items()
                )
            ),
            "target_element_histogram": dict(
                sorted(
                    Counter(nx.get_node_attributes(target_heavy, "element").values()).items()
                )
            ),
        },
    )
    return analyse_hydrogens(reference, target, result)


def reindex(
    reference_path: str | Path,
    target_path: str | Path,
    output_path: str | Path,
    *,
    tol_bond: float = 0.45,
    optimise: bool = False,
    orcadir: str | Path | None = None,
    charge: int | None = None,
    multiplicity: int | None = None,
    timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> ReindexResult:
    """Run graph reindexing, hydrogen generation, and optional ORCA optimization."""
    reference_path = Path(reference_path)
    target_path = Path(target_path)
    output_path = Path(output_path)
    if optimise and (orcadir is None or charge is None or multiplicity is None):
        raise ValueError(
            "optimise=True requires orcadir, charge, and multiplicity"
        )
    reference_df = pdb2df(reference_path, logger=logger)
    target_df = pdb2df(target_path, logger=logger)
    if reference_df is None:
        raise ValueError(f"Failed to read reference PDB: {reference_path}")
    if target_df is None:
        raise ValueError(f"Failed to read target PDB: {target_path}")

    reference_graph = pdb2graph(reference_path, tol_bond=tol_bond, logger=logger)
    target_graph = pdb2graph(target_path, tol_bond=tol_bond, logger=logger)
    result = reindex_graphs(reference_graph, target_graph)
    result = generate_missing_hydrogens(reference_graph, target_graph, result)
    write_reindexed_pdb(
        reference_df,
        target_df,
        result,
        output_path,
        reference_graph=reference_graph,
        target_graph=target_graph,
        logger=logger,
    )

    result.validation["output_path"] = str(output_path)
    result.validation["hydrogen_optimisation_requested"] = bool(optimise)
    if optimise:
        artifacts = optimise_reindexed_pdb(
            reference_path,
            output_path,
            orcadir,
            charge,
            multiplicity,
            timeout=timeout,
            logger=logger,
        )
        result.validation.update(
            {
                "hydrogen_optimisation_workdir": str(artifacts.workdir),
                "hydrogen_optimisation_staged_pdb": str(artifacts.staged_pdb),
                "hydrogen_optimisation_input": str(artifacts.input_path),
                "hydrogen_optimisation_stdout": str(artifacts.stdout_path),
                "hydrogen_optimisation_stderr": str(artifacts.stderr_path),
                "hydrogen_optimisation_xyz": str(artifacts.optimized_xyz),
                "hydrogen_optimisation_labelled_pdb": str(artifacts.labelled_pdb),
                "hydrogen_optimisation_return_code": artifacts.return_code,
            }
        )
        result.validation["hydrogen_optimisation_completed"] = True
    else:
        result.validation["hydrogen_optimisation_completed"] = False
    return result
