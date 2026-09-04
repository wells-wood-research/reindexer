"""Backward-compatible wrapper for the reindexer package.

Use ``python -m reindexer`` or the installed ``reindexer`` command for new
code.  Public functions remain available from this module for existing users.
"""

from reindexer.reindex import *  # noqa: F401,F403
from reindexer.cli import main as cli_main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
