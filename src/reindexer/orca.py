"""Minimal ORCA integration required for hydrogen optimization."""

import os
from pathlib import Path
import subprocess
from typing import Optional
import logging


def write_hydrogen_opt(
    inp_file: str | Path,
    pdb_file: str | Path,
    charge: int,
    multiplicity: int,
) -> Path:
    """Write the ORCA input used to optimize hydrogen positions."""
    inp_file = Path(inp_file)
    inp_file.parent.mkdir(parents=True, exist_ok=True)
    pdb_file = Path(pdb_file).resolve()
    inp_file.write_text(
        "! XTB2 OptH\n"
        f"*pdbfile {int(charge)} {int(multiplicity)} {pdb_file}\n"
    )
    return inp_file


def run_orca(
    input_path: str | Path,
    orcadir: str | Path,
    timeout: Optional[float],
    logger: Optional[logging.Logger] = None,
) -> int:
    """Run ORCA and persist stdout/stderr beside its input file."""
    input_path = Path(input_path)
    orcadir = Path(orcadir)
    stdout_path = input_path.with_suffix(".out")
    stderr_path = input_path.with_suffix(".err")
    orca_executable = orcadir / "orca"
    xtb_executable = orcadir / "otool_xtb"
    environment = {
        "PATH": f"{orcadir}:{os.environ.get('PATH', '')}",
        "XTBEXE": str(xtb_executable),
        "HOME": os.environ.get("HOME", ""),
        "USER": os.environ.get("USER", ""),
        "TMPDIR": "/tmp",
        "OMPI_MCA_hwloc_base_binding_policy": "none",
    }
    try:
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            completed = subprocess.run(
                [str(orca_executable), str(input_path)],
                check=False,
                timeout=timeout,
                stdout=stdout,
                stderr=stderr,
                env=environment,
            )
    except subprocess.TimeoutExpired:
        if logger:
            logger.error("ORCA exceeded %s seconds and was terminated", timeout)
        return 1
    except Exception as exc:
        if logger:
            logger.error("Unexpected exception during ORCA run: %s", exc)
        return 1
    return completed.returncode
