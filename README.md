# reindexer

`reindexer` matches the atoms of a target molecular structure to a reference
structure and writes the target coordinates in reference atom order. The
current implementation is a Python library. It has no command-line entry
point.

## Installation

The package requires Python 3.10 or newer.

The distribution is named `molecular-reindexer`; the import package is
`reindexer`.

```bash
pip install -e .
```

Install the test dependencies with:

```bash
pip install -e ".[test]"
```

## Public API

The main entry point is `reindexer.reindex`:

```python
from reindexer import reindex

result = reindex(
    "reference.pdb",
    "target.pdb",
    "target_reindexed.pdb",
)
```

The arguments are:

```python
reindex(
    reference_path,
    target_path,
    output_path,
    *,
    tol_bond=0.45,
    optimise=False,
    orcadir=None,
    charge=None,
    multiplicity=None,
    timeout=None,
    logger=None,
    allow_subgraph=False,
)
```

`reference_path`, `target_path`, and `output_path` may be strings or
`pathlib.Path` objects. Parent directories for the output are created
automatically. The function returns a `ReindexResult` after writing the PDB.

Matching is strict by default. Set `allow_subgraph=True` when the reference is
an atom fragment that may be contained in a larger target structure, such as an
amino-acid side chain extracted from a protein model. The target must contain
at least as many heavy atoms as the reference; target-only heavy atoms are
placed after the matched reference atoms in the output.

`reindex_graphs(reference, target, allow_subgraph=False)` is also exported. It accepts NetworkX
graphs produced by `reindexer.graph` and performs heavy-atom matching plus
hydrogen analysis without writing a file or generating missing coordinates.
Use `generate_missing_hydrogens` and `write_reindexed_pdb` from their modules
for those lower-level steps.

## Processing Model

The full `reindex` pipeline is:

1. Parse both PDB files into pandas DataFrames.
2. Build distance-inferred NetworkX graphs.
3. Remove hydrogens and match the heavy-atom graphs.
4. Assign existing target hydrogens to reference hydrogen slots.
5. Generate coordinates for missing reference hydrogens.
6. Build and validate a reference-ordered PDB, or, in subgraph mode, a PDB
   with matched target atoms first and target-only atoms appended.
7. Optionally optimize hydrogen coordinates with ORCA.

Heavy-atom mappings are one-to-one and use zero-based internal graph indices.
The result also contains the inverse mapping, mapping scores, hydrogen slot
assignments, generated hydrogen placements, and validation diagnostics.

## Input And Output

The end-to-end pipeline reads PDB files containing `ATOM` or `HETATM`
records. The parser reads fixed-width atom fields for:

- record type
- atom serial
- atom name
- residue name and number
- chain identifier
- coordinates
- occupancy
- B-factor
- element

The element field is inferred from the atom name when it is empty. Non-atom
records, including headers, connectivity records, and model delimiters, are
ignored. The parser therefore does not interpret PDB connectivity records and
should be given one intended structure. Coordinates must be finite and atom
records must be well formed.

In strict mode, output rows are exactly the reference rows and remain in
reference order. Reference atom names, residue fields, element fields,
serials, occupancy, and B-factors are retained. In subgraph mode, matched
target atoms are ordered according to their reference atom indices and
unmatched target atoms are appended after them. Heavy-atom coordinates and
coordinates of retained target hydrogens come from the target. PDB output
coordinates are serialized to three decimal places.

Target hydrogens that are not needed to fill reference hydrogen slots are
omitted from the output. If a target parent has fewer hydrogens than its
reference counterpart, the missing reference slots are generated. Unattached
or multiply attached target hydrogens are rejected as invalid topology.

## Connectivity And Matching

Connectivity is inferred from pairwise distances. An edge is created when:

```text
distance < covalent_radius(atom_a) + covalent_radius(atom_b) + tol_bond
```

The default `tol_bond` is `0.45` angstrom. No bond order, formal charge, or
valence model is used. Supported covalent radii are defined in
`reindexer.graph`; unknown elements use a fallback radius of `0.6` angstrom.

In strict mode, heavy-atom graph candidates are resolved in this order:

1. connectivity signatures containing element and local degree information
2. total bond-distance difference
3. local-neighbor angle difference
4. Kabsch-aligned heavy-atom RMSD

`GraphMismatchError` is raised when the heavy graphs cannot be matched.
`MappingAmbiguityError` is raised when more than one candidate remains tied
after all scoring stages. Candidate and score diagnostics are included in
these exceptions.

With `allow_subgraph=True`, the reference heavy graph must be no larger than
the target heavy graph. Subgraph matching finds candidate mappings using
element identity, then the bond-distance, local-angle, and aligned RMSD
tie-breakers are applied to the matched subgraph. Missing reference atoms and
ambiguous matches still raise errors.

## Hydrogen Generation

Hydrogen generation uses target heavy-atom geometry and deterministic templates;
it does not use RDKit or reference hydrogen coordinates. The implemented
geometry kinds are terminal, linear, tetrahedral, trigonal planar, and
trigonal pyramidal. Bond lengths are defined for C, N, O, P, and S, with a
`1.00` angstrom fallback for other elements.

Generated atoms are represented by `HydrogenPlacement` objects containing the
reference hydrogen index, target parent index, coordinates, geometry kind, and
bond length.

## Optional ORCA Optimization

Set `optimise=True` to run the optional ORCA hydrogen optimization stage:

```python
result = reindex(
    "reference.pdb",
    "target.pdb",
    "target_reindexed.pdb",
    optimise=True,
    orcadir="/path/to/orca",
    charge=0,
    multiplicity=1,
    timeout=3600,
)
```

When optimization is enabled, `orcadir`, `charge`, and `multiplicity` are
required. The directory must contain the ORCA executable and its `otool_xtb`
helper. The stage writes persistent files beside the output in a directory
named `<output-stem>_orca`, including `opt.inp`, `opt.out`, `opt.err`, and
`opt.xyz`. The XYZ result is checked against the reference atom count and
element order, labelled with reference PDB metadata, and copied over the
reindexed output. The optimized XYZ/PDB are not subjected to an additional
heavy-atom coordinate validation after ORCA completes.

## Lower-Level Utilities

The modules also provide:

- `reindexer.io`: PDB DataFrame conversion and strict XYZ reading
- `reindexer.graph`: DataFrame, PDB, and XYZ graph construction
- `reindexer.matching`: heavy-atom matching and matching exceptions
- `reindexer.hydrogens`: hydrogen analysis and coordinate generation
- `reindexer.writer`: in-memory PDB construction and validation
- `reindexer.labelling`: apply XYZ coordinates to reference PDB labels
- `reindexer.orca`: ORCA input generation and subprocess execution

There is no `reindexer` CLI and no `python -m reindexer` entry point in the
current package.

## Tests

After installing the package and test dependencies, run:

```bash
pytest -q
```
