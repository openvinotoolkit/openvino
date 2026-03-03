# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import sys

COVERAGE_ROOT = Path(__file__).resolve().parents[2]
if str(COVERAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(COVERAGE_ROOT))
