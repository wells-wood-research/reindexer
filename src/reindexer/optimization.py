"""Optional ORCA-based hydrogen optimization after reindexing."""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional
import logging
import shutil

from . import orca
from .labelling import label_xyz_with_reference


class HydrogenOptimizationError(RuntimeError):
    """Raised when optional ORCA hydrogen optimization cannot complete."""

    def __init__(self, diagnostics: dict[str, Any]):
        self.diagnostics = diagnostics
        super().__init__(f"Hydrogen optimization failed: {diagnostics}")


@dataclass(frozen=True)
class HydrogenOptimizationArtifacts:
    """Persistent files produced by one optional ORCA optimization run."""

    workdir: Path
    staged_pdb: Path
    input_path: Path
    stdout_path: Path
    stderr_path: Path
    optimized_xyz: Path
    labelled_pdb: Path
    return_code: int


def _artifacts(reindexed_path: Path) -> HydrogenOptimizationArtifacts:
    workdir = reindexed_path.parent / f"{reindexed_path.stem}_orca"
    return HydrogenOptimizationArtifacts(
        workdir=workdir,
        staged_pdb=workdir / "reindexed.pdb",
        input_path=workdir / "opt.inp",
        stdout_path=workdir / "opt.out",
        stderr_path=workdir / "opt.err",
        optimized_xyz=workdir / "opt.xyz",
        labelled_pdb=workdir / "optimized_reindexed.pdb",
        return_code=-1,
    )


def optimise_reindexed_pdb(
    reference_path: str | Path,
    reindexed_path: str | Path,
    orcadir: str | Path,
    charge: int,
    multiplicity: int,
    timeout: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> HydrogenOptimizationArtifacts:
    """Optimize hydrogen coordinates using persistent files beside output.

    ``orcadir`` is an ORCA installation directory containing the ``orca``
    executable. No heavy-atom coordinate validation is performed after ORCA.
    """
    reference_path = Path(reference_path)
    reindexed_path = Path(reindexed_path)
    orcadir = Path(orcadir)
    if not reference_path.is_file():
        raise HydrogenOptimizationError(
            {"reason": "reference PDB does not exist", "path": str(reference_path)}
        )
    if not reindexed_path.is_file():
        raise HydrogenOptimizationError(
            {"reason": "reindexed PDB does not exist", "path": str(reindexed_path)}
        )
    if not orcadir.is_dir():
        raise HydrogenOptimizationError(
            {"reason": "orcadir must be an installation directory", "path": str(orcadir)}
        )

    artifacts = _artifacts(reindexed_path)
    artifacts.workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reindexed_path, artifacts.staged_pdb)

    orca.write_hydrogen_opt(
        artifacts.input_path,
        artifacts.staged_pdb,
        charge,
        multiplicity,
    )
    return_code = orca.run_orca(
        artifacts.input_path,
        orcadir,
        timeout,
        logger=logger,
    )
    artifacts = replace(artifacts, return_code=return_code)
    if return_code != 0:
        raise HydrogenOptimizationError(
            {
                "reason": "orca returned a nonzero status",
                "return_code": return_code,
                "input_path": str(artifacts.input_path),
                "stdout_path": str(artifacts.stdout_path),
                "stderr_path": str(artifacts.stderr_path),
            }
        )
    if not artifacts.optimized_xyz.is_file():
        raise HydrogenOptimizationError(
            {
                "reason": "orca did not produce opt.xyz",
                "path": str(artifacts.optimized_xyz),
                "stdout_path": str(artifacts.stdout_path),
                "stderr_path": str(artifacts.stderr_path),
            }
        )

    try:
        label_xyz_with_reference(
            reference_path,
            artifacts.optimized_xyz,
            artifacts.labelled_pdb,
        )
    except Exception as exc:
        raise HydrogenOptimizationError(
            {
                "reason": "failed to label ORCA XYZ with reference PDB",
                "xyz_path": str(artifacts.optimized_xyz),
                "labelled_pdb": str(artifacts.labelled_pdb),
                "error": str(exc),
            }
        ) from exc

    shutil.copy2(artifacts.labelled_pdb, reindexed_path)
    return artifacts
