#!/usr/bin/env python3
"""Run the Analyze and Convert agent.

Probes the model, tries a systematic conversion strategy matrix
(fp16/int8/int4 × stable/git-HEAD), classifies every failure with full
traceback analysis, and builds a structured diagnostic report with routing
signals for the orchestrator.

Usage:
    python .github/scripts/meat/run_analyze_and_convert.py <context-file>

Run from the openvino repo root.

Minimal context file:

    Model: Qwen/Qwen3-0.6B

Output goes to agent-results/analyze-and-convert/.

Copilot CLI reference:
  https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference
"""

import os
import subprocess
import sys

AGENT_FILE = ".github/agents/analyze-and-convert.agent.md"


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

    output_dir = "agent-results/analyze-and-convert"
    os.makedirs(output_dir, exist_ok=True)

    with open(context_file_path) as f:
        prompt = f.read()

    cmd = [
        "copilot",
        "--agent", "analyze-and-convert",
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
