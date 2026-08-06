# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Entry point for operator-level enablement (OpenVINO core pipeline).

Passes the context file to the `enable-operator` agent defined in
`.github/agents-prototype/enable-operator.agent.md`, which drives the full
FE → Core OpSpec → Transformation/CPU/GPU/NPU → Package Builder pipeline.

Usage:
    python .github/scripts/meat/enable_operator.py <context-file>

Run from the openvino repo root.

The context file should contain:
  - the operator name (e.g. ``aten::erfinv``),
  - a model ID or local path to a model that uses the operator — the agents
    will use this model to test and validate the implementation.

Example context file:

    Operator: aten::erfinv
    Model: Qwen/Qwen3-0.6B
    Error: No conversion rule for aten::erfinv

    erfinv computes the element-wise inverse error function.
    PyTorch docs: https://pytorch.org/docs/stable/generated/torch.erfinv.html

See .github/scripts/meat/README.md for the full context file format.

Shortcut for: run_agent.py enable-operator <context-file>
"""

import os
import subprocess
import sys


def main() -> None:
    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_agent.py")
    sys.exit(subprocess.run([sys.executable, runner, "enable-operator"] + sys.argv[1:]).returncode)


if __name__ == "__main__":
    main()
