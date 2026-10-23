# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest


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
            expected["math"]["sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                runtime.record_identity(expected, output)
            self.assertFalse(json.loads(output.read_text())["passed"])
        self.assertIsNotNone(library)


if __name__ == "__main__":
    unittest.main()
