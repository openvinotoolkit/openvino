# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("runtime_identity", Path(__file__).parents[1] / "runtime_identity.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


@unittest.skipUnless(Path("/proc/self/maps").is_file(), "Linux process maps required")
class RuntimeIdentityTests(unittest.TestCase):
    def test_cli_preserves_expected_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expected, alias = path / "expected.json", path / "alias.json"
            expected.write_text('{}')
            os.link(expected, alias)
            for output in (expected, alias):
                result = subprocess.run([sys.executable, runtime.__file__, "--expected", str(expected),
                                         "--output", str(output)], capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(expected.read_text(), '{}')

    def test_report_preserves_expected_library_aliases_with_invalid_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            library = path / "library.so"
            original = b"KEEP LIBRARY BYTES"
            library.write_bytes(original)
            symlink, hardlink = path / "symlink.json", path / "hardlink.json"
            symlink.symlink_to(library)
            os.link(library, hardlink)
            expected = {"fixture": {"path": str(library), "sha256": "invalid"}}
            for output in (library, symlink, hardlink):
                with self.subTest(output=output):
                    with self.assertRaisesRegex(ValueError, "must not overwrite"):
                        runtime.record_identity(expected, output)
                    self.assertEqual(library.read_bytes(), original)

    def test_report_preserves_mapped_library_not_in_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            mapped, expected_library = path / "libopenvino_fixture.so", path / "expected.so"
            original = b"KEEP MAPPED LIBRARY BYTES"
            mapped.write_bytes(original)
            expected_library.write_bytes(b"EXPECTED")
            expected = {"fixture": {"path": str(expected_library),
                                    "sha256": hashlib.sha256(expected_library.read_bytes()).hexdigest()}}
            with mapped.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ):
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    runtime.record_identity(expected, mapped)
            self.assertEqual(mapped.read_bytes(), original)

    def test_report_preserves_mapped_aliases_after_manifest_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            mapped, expected_library = path / "libopenvino_fixture.so", path / "expected.so"
            original = b"KEEP MAPPED LIBRARY BYTES"
            mapped.write_bytes(original)
            expected_library.write_bytes(b"EXPECTED")
            symlink, hardlink = path / "symlink.json", path / "hardlink.json"
            symlink.symlink_to(mapped)
            os.link(mapped, hardlink)
            manifests = (
                {"fixture": {"path": str(expected_library), "sha256": "invalid"}},
                {"fixture": {"path": str(path / "missing.so"), "sha256": "0" * 64}},
                {},
            )
            with mapped.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ):
                for expected in manifests:
                    for output in (mapped, symlink, hardlink):
                        with self.subTest(expected=expected, output=output):
                            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                                runtime.record_identity(expected, output)
                            self.assertEqual(mapped.read_bytes(), original)

    def test_hash_failure_keeps_all_mappings_and_protects_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            first, second = path / "libopenvino_first.so", path / "libopenvino_second.so"
            first.write_bytes(b"FIRST")
            second.write_bytes(b"SECOND")
            expected = {"fixture": {"path": str(first), "sha256": "0" * 64}}
            output = path / "report.json"
            read_bytes = Path.read_bytes

            def fail_first_hash(library):
                if library == first:
                    raise OSError("fixture hash failure")
                return read_bytes(library)

            with first.open("rb") as stream1, second.open("rb") as stream2, \
                    mmap.mmap(stream1.fileno(), 0, access=mmap.ACCESS_READ), \
                    mmap.mmap(stream2.fileno(), 0, access=mmap.ACCESS_READ), \
                    mock.patch.object(Path, "read_bytes", fail_first_hash):
                with self.assertRaisesRegex(RuntimeError, "fixture hash failure"):
                    runtime.record_identity(expected, output)
                report = json.loads(output.read_text())
                self.assertFalse(report["passed"])
                self.assertTrue({str(first), str(second)}.issubset(
                    {entry["path"] for entry in report["mapped_libraries"]}
                ))
                with self.assertRaisesRegex(ValueError, "must not overwrite"):
                    runtime.record_identity(expected, second)
            self.assertEqual(first.read_bytes(), b"FIRST")
            self.assertEqual(second.read_bytes(), b"SECOND")

    def test_mapping_discovery_failure_refuses_report_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with mock.patch.object(Path, "read_text", side_effect=OSError("fixture maps failure")):
                with self.assertRaisesRegex(RuntimeError, "refusing report output"):
                    runtime.record_identity({}, output)
            self.assertFalse(output.exists())

    def test_actual_mapped_library_and_wrong_hash(self):
        library = ctypes.CDLL("libm.so.6")
        paths = [Path(line.split(maxsplit=5)[5]) for line in Path("/proc/self/maps").read_text().splitlines()
                 if len(line.split(maxsplit=5)) == 6 and line.split(maxsplit=5)[5].startswith("/")
                 and Path(line.split(maxsplit=5)[5]).name.startswith("libm.so")]
        self.assertTrue(paths)
        mapped = paths[0].resolve()
        expected = {"math": {"path": str(mapped), "sha256": hashlib.sha256(mapped.read_bytes()).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "identity.json"
            self.assertTrue(runtime.record_identity(expected, output)["passed"])
            alias = Path(directory) / "math-alias.so"
            alias.symlink_to(mapped)
            expected["math"]["path"] = str(alias)
            self.assertTrue(runtime.record_identity(expected, output)["passed"])
            expected["math"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                runtime.record_identity(expected, output)
            self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertIsNotNone(library)


if __name__ == "__main__":
    unittest.main()
