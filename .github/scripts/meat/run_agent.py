# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""Run any OpenVINO coding agent from .github/agents-prototype/.

Usage:
    python .github/scripts/meat/run_agent.py <agent-name> <context-file>
    python .github/scripts/meat/run_agent.py --list

Run from the openvino repo root.

Output goes to agent-results/<agent-name>/ in the working directory:
  session.md           — full agent session transcript
  patches/             — generated .patch files ready to apply
  pipeline_state.json  — shared state read/written by all agents

Copilot CLI reference:
  https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
"""

import os
import subprocess
import sys

AGENTS = [
    "enable-operator",
    "frontend",
    "core-opspec",
    "transformation",
    "cpu",
    "gpu",
    "deployer",
    "analyze-and-convert",
]

_WARN = (
    "\n"
    "  ╔══════════════════════════════════════════════════════════════════╗\n"
    "  ║  WARNING: AUTONOMOUS / UNATTENDED MODE                         ║\n"
    "  ║                                                                ║\n"
    "  ║  This script runs GitHub Copilot with --no-ask-user and        ║\n"
    "  ║  --autopilot.  The agent will READ, CREATE, and MODIFY files   ║\n"
    "  ║  in this repository WITHOUT asking for confirmation.           ║\n"
    "  ║                                                                ║\n"
    "  ║  Review agent-results/ after the run and apply patches with   ║\n"
    "  ║  'git apply' — do NOT blindly commit generated changes.        ║\n"
    "  ╚══════════════════════════════════════════════════════════════════╝\n"
)


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] in ("--list", "-l"):
        print("Available agents:")
        for name in AGENTS:
            print(f"  {name}")
        sys.exit(0)

    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <agent-name> <context-file>\n"
            f"       {sys.argv[0]} --list",
            file=sys.stderr,
        )
        sys.exit(1)

    agent_name = sys.argv[1]
    context_file_path = sys.argv[2]
    agent_file = f".github/agents-prototype/{agent_name}.agent.md"

    if not os.path.isfile(context_file_path):
        print(f"Error: context file not found: {context_file_path}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(agent_file):
        print(
            f"Error: agent file not found: {agent_file}\n"
            "Make sure you are running from the openvino repo root.\n"
            f"Available agents: {', '.join(AGENTS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = f"agent-results/{agent_name}"
    os.makedirs(output_dir, exist_ok=True)

    with open(context_file_path) as f:
        prompt = f.read()

    print(_WARN, flush=True)

    cmd = [
        "copilot",
        "--agent", agent_name,
        "--share", f"{output_dir}/session.md",
        "--no-ask-user",
        "--autopilot",
        "--stream", "on",
        "--log-level", "all",
        "-p", prompt,
    ]

    sys.exit(subprocess.run(cmd, shell=(sys.platform == "win32")).returncode)


if __name__ == "__main__":
    main()
