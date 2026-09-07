import importlib.util
import tempfile
import unittest
from pathlib import Path

from reindexer import (
    PDBFormatError,
    UnsupportedPDBFeatureError,
    default_output_path,
    parse_pdb,
    reindex_pdb,
)


def atom_line(serial, name, element, x, y, z, residue="LIG", chain="A", residue_id=1):
    return (
        f"{'HETATM':<6}{serial:5d} {name:>4}{'':1}{residue:>3} {chain:1}"
        f"{residue_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}"
        f"{1.0:6.2f}{12.5:6.2f}          {element:>2}  "
    )


def water_pdb(order, coordinates, serials):
    atoms = {
        "O1": ("O", coordinates["O1"]),
        "H2": ("H", coordinates["H2"]),
        "H3": ("H", coordinates["H3"]),
    }
    lines = [atom_line(serials[name], name, atoms[name][0], *atoms[name][1]) for name in order]
    lines.extend(
        [
            f"CONECT{serials['O1']:5d}{serials['H2']:5d}{serials['H3']:5d}",
            f"CONECT{serials['H2']:5d}{serials['O1']:5d}",
            f"CONECT{serials['H3']:5d}{serials['O1']:5d}",
            "END",
        ]
    )
    return "\n".join(lines) + "\n"


class PDBParserTests(unittest.TestCase):
    def test_default_output_path_is_a_target_sibling(self):
        self.assertEqual(
            default_output_path("structures/target.pdb"),
            Path("structures/target_reindexed.pdb"),
        )

    def test_parser_retains_atom_fields_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pdb"
            path.write_text(
                "HEADER    TEST\n"
                + atom_line(1, "C1", "C", 1.234, 2.345, 3.456)
                + "\nREMARK    retained\nEND\n"
            )
            document = parse_pdb(path)

        self.assertEqual(document.atoms[0].serial, 1)
        self.assertEqual(document.atoms[0].atom_name, "C1")
        self.assertEqual(document.atoms[0].element, "C")
        self.assertEqual(document.metadata_lines, ("HEADER    TEST", "REMARK    retained"))

    def test_parser_rejects_unsupported_alternate_location(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pdb"
            line = atom_line(1, "C1", "C", 0, 0, 0)
            line = line[:16] + "B" + line[17:]
            path.write_text(line + "\nEND\n")
            with self.assertRaises(UnsupportedPDBFeatureError):
                parse_pdb(path)

    def test_parser_rejects_short_atom_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pdb"
            path.write_text("ATOM      1  C1  LIG A   1\n")
            with self.assertRaises(PDBFormatError):
                parse_pdb(path)

    def test_serial_policy_is_validated_before_loading(self):
        with self.assertRaises(ValueError):
            reindex_pdb("missing-reference.pdb", "missing-target.pdb", "output.pdb", serial_policy="input")


@unittest.skipUnless(importlib.util.find_spec("rdkit"), "RDKit is required")
class PDBReindexTests(unittest.TestCase):
    def _write_inputs(self, directory):
        reference = Path(directory) / "reference.pdb"
        target = Path(directory) / "target.pdb"
        coordinates = {"O1": (10.0, 11.0, 12.0), "H2": (13.0, 14.0, 15.0), "H3": (16.0, 17.0, 18.0)}
        reference.write_text(
            water_pdb(
                ("O1", "H2", "H3"),
                {"O1": (0, 0, 0), "H2": (0, 1, 0), "H3": (1, 0, 0)},
                {"O1": 1, "H2": 2, "H3": 3},
            )
        )
        target.write_text(
            water_pdb(
                ("H3", "O1", "H2"),
                coordinates,
                {"O1": 10, "H2": 20, "H3": 30},
            )
        )
        return reference, target, coordinates

    def test_reference_serial_policy_preserves_target_coordinates_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, target, coordinates = self._write_inputs(directory)
            output = Path(directory) / "output.pdb"
            result = reindex_pdb(reference, target, output, serial_policy="reference")
            document = parse_pdb(output)

        self.assertEqual([atom.serial for atom in document.atoms], [1, 2, 3])
        self.assertEqual([atom.atom_name for atom in document.atoms], ["O1", "H2", "H3"])
        self.assertEqual((document.atoms[0].x, document.atoms[0].y, document.atoms[0].z), coordinates["O1"])
        self.assertEqual(result.reference_to_target, {1: 10, 2: 20, 3: 30})

    def test_target_serial_policy_keeps_existing_target_serials(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, target, _ = self._write_inputs(directory)
            output = Path(directory) / "output.pdb"
            reindex_pdb(reference, target, output, serial_policy="target")
            document = parse_pdb(output)

        self.assertEqual([atom.serial for atom in document.atoms], [10, 20, 30])
        self.assertEqual([atom.atom_name for atom in document.atoms], ["O1", "H2", "H3"])

    def test_missing_reference_hydrogens_are_generated(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _ = self._write_inputs(directory)
            target = Path(directory) / "target_missing_hydrogens.pdb"
            target.write_text(
                atom_line(10, "O1", "O", 10.0, 11.0, 12.0) + "\nEND\n"
            )
            output = Path(directory) / "output.pdb"
            result = reindex_pdb(reference, target, output)
            document = parse_pdb(output)

        self.assertEqual(result.generated_hydrogens, (2, 3))
        self.assertEqual(len(document.atoms), 3)
        self.assertTrue(all(atom.serial in {1, 2, 3} for atom in document.atoms))
        self.assertTrue(any((atom.x, atom.y, atom.z) != (0.0, 1.0, 0.0) for atom in document.atoms[1:]))

    def test_target_only_hydrogens_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            reference, _, _ = self._write_inputs(directory)
            target = Path(directory) / "target_extra_hydrogen.pdb"
            target.write_text(
                "\n".join(
                    [
                        atom_line(10, "O1", "O", 10.0, 11.0, 12.0),
                        atom_line(20, "H2", "H", 13.0, 14.0, 15.0),
                        atom_line(30, "H3", "H", 16.0, 17.0, 18.0),
                        atom_line(40, "H4", "H", 19.0, 20.0, 21.0),
                        "END",
                    ]
                )
                + "\n"
            )
            output = Path(directory) / "output.pdb"
            result = reindex_pdb(reference, target, output)

        self.assertEqual(result.removed_target_hydrogens, (40,))
        self.assertEqual(len(parse_pdb(output).atoms), 3)
