"""Command-line interface for reindexer."""

from __future__ import annotations

import argparse
from typing import Optional, Sequence

def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate a reindexed structure file with atom order matched to "
            "the reference input."
        )
    )
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to the reference structure whose atom order is canonical.",
    )
    parser.add_argument(
        "--referee",
        "--target",
        dest="target",
        required=True,
        help="Path to the target structure to reorder.",
    )
    parser.add_argument(
        "--outDir",
        required=True,
        help="Directory to which output files will be written.",
    )
    parser.add_argument(
        "--serial-policy",
        choices=("reference", "target"),
        default="reference",
        help="PDB output serial-number policy (default: reference).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the CLI and return a process exit status."""

    args = build_parser().parse_args(argv)
    try:
        from .reindex import main as reindex_main

        reindex_main(
            args.reference,
            args.target,
            args.outDir,
            serial_policy=args.serial_policy,
        )
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
