#!/usr/bin/env python3
"""Entry point for operator-level enablement (OpenVINO core pipeline).

Passes the context file to the `enable-operator` agent defined in
`.github/agents/enable-operator.agent.md`, which drives the full
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
"""

import os
import subprocess
import sys

AGENT_FILE = ".github/agents/enable-operator.agent.md"


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

    output_dir = "agent-results/enable-operator"
    os.makedirs(output_dir, exist_ok=True)

    with open(context_file_path) as f:
        prompt = f.read()

    cmd = [
        "copilot",
        "--agent", "enable-operator",
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
