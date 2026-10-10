# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import subprocess
import sys

import pytest

transformers = pytest.importorskip("transformers")
if int(transformers.__version__.split(".")[0]) < 5:
    pytest.skip("Install transformers_tests/requirements.txt", allow_module_level=True)


@pytest.mark.precommit
def test_coverage_inventory(tmp_path):
    suite = Path(__file__).resolve().parent
    result = subprocess.run(
        [sys.executable, str(suite / "scripts" / "op_coverage.py"),
         "--output", str(tmp_path / "op-coverage.json"),
         "--check-baseline", str(suite / "coverage_inventory.json")],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
