# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Run the Core OpSpec agent.

Implements a new operator in OpenVINO core: C++ class definition,
shape inference, opset registration, reference kernel, and RST
documentation. Invoked after the FE agent escalates with
`status=escalate_to_core`.

Usage:
    python .github/scripts/meat/run_core_opspec.py <context-file>

Run from the openvino repo root.

Example context file:

    Operator: aten::erfinv
    Model: Qwen/Qwen3-0.6B

    FE agent escalated — no existing OV op covers erfinv semantics.
    Escalation payload:
      op_name: aten::erfinv
      op_semantics: element-wise inverse error function, single float tensor in/out
      suggested_ov_decomposition: null

Output goes to agent-results/core-opspec/.

Copilot CLI reference:
  https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
"""

import os
import subprocess
import sys

AGENT_FILE = ".github/agents/core-opspec.agent.md"


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

    output_dir = "agent-results/core-opspec"
    os.makedirs(output_dir, exist_ok=True)

    with open(context_file_path) as f:
        prompt = f.read()

    cmd = [
        "copilot",
        "--agent", "core-opspec",
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
