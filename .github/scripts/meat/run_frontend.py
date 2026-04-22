# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Run the Frontend agent.

Writes OpenVINO frontend conversion rules for a framework operator
(PyTorch, ONNX, TensorFlow → OV graph nodes). Invoked when the Deployer or
Analyze-and-Convert agent classifies the failure as `missing_conversion_rule`
or `frontend_error`.

Usage:
    python .github/scripts/meat/run_frontend.py <context-file>

Run from the openvino repo root.

Example context file:

    Model: Qwen/Qwen3-0.6B
    Operator: aten::erfinv
    Error traceback:
      RuntimeError: No conversion rule found for aten::erfinv
      at openvino/frontend/pytorch/ts_decoder.py:287

    erfinv computes the element-wise inverse error function.
    PyTorch docs: https://pytorch.org/docs/stable/generated/torch.erfinv.html

Output goes to agent-results/frontend/.

Copilot CLI reference:
  https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
"""

import os
import subprocess
import sys

AGENT_FILE = ".github/agents/frontend.agent.md"


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <context-file>", file=sys.stderr)
        sys.exit(1)

    context_file_path = sys.argv[1]

    if not os.path.isfile(context_file_path):
        print(f"Error: context file not found: {context_file_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(AGENT_FILE):
        print(
            f"Error: agent file not found: {AGENT_FILE}\n"
            "Make sure you are running from the openvino repo root.",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = "agent-results/frontend"
    os.makedirs(output_dir, exist_ok=True)

    with open(context_file_path) as f:
        prompt = f.read()

    cmd = [
        "copilot",
        "--agent", "frontend",
        "--share", f"{output_dir}/session.md",
        "--allow-all",
        "--no-ask-user",
        "--autopilot",
        "--stream", "on",
        "--log-level", "all",
        "-p", prompt,
    ]

    sys.exit(subprocess.run(cmd, shell=(sys.platform == "win32")).returncode)


if __name__ == "__main__":
    main()
