#!/usr/bin/env python3
"""Set up the environment needed to run agent scripts from .github/scripts/meat/.

Checks and (where possible) performs:
  1. Python version requirement (3.10+)
  2. GitHub CLI (gh) installation
  3. copilot CLI installation
  4. copilot authentication
  5. Python package dependencies (requirements.txt in repo root, if present)

Run from the openvino repo root:
  python .github/scripts/meat/setup.py
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent.parent
REQUIREMENTS = REPO_ROOT / "requirements.txt"
MIN_PYTHON = (3, 10)


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def ok(msg: str) -> None:
    print(f"  [OK]  {msg}")


def info(msg: str) -> None:
    print(f"  [..] {msg}")


def warn(msg: str) -> None:
    print(f"  [!!] {msg}", file=sys.stderr)


def fail(msg: str) -> None:
    print(f"\n  [FAIL] {msg}\n", file=sys.stderr)
    sys.exit(1)


def run(*args, **kwargs):
    return subprocess.run(list(args), **kwargs)


# ──────────────────────────────────────────────
# Python
# ──────────────────────────────────────────────

def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        fail(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, found {sys.version.split()[0]}")
    ok(f"Python {sys.version.split()[0]}")


# ──────────────────────────────────────────────
# GitHub CLI (gh)
# ──────────────────────────────────────────────

def _gh_version() -> Optional[str]:
    if not shutil.which("gh"):
        return None
    try:
        r = run("gh", "--version", capture_output=True, text=True, timeout=10)
        line = (r.stdout or "").splitlines()
        return line[0].strip() if line else "unknown"
    except Exception:
        return None


def _try_install_gh() -> bool:
    system = platform.system()
    if system == "Windows":
        if shutil.which("winget"):
            info("Attempting: winget install --id GitHub.cli --silent …")
            r = run("winget", "install", "--id", "GitHub.cli", "--silent",
                    "--accept-package-agreements", "--accept-source-agreements")
            return r.returncode == 0
    elif system == "Darwin":
        if shutil.which("brew"):
            info("Attempting: brew install gh …")
            r = run("brew", "install", "gh")
            return r.returncode == 0
    else:  # Linux
        if shutil.which("apt-get") and shutil.which("sudo"):
            info("Attempting apt-based gh install …")
            cmds = [
                ["sudo", "mkdir", "-p", "/etc/apt/keyrings"],
                ["sudo", "bash", "-c",
                 "curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg"
                 " -o /etc/apt/keyrings/githubcli-archive-keyring.gpg"],
                ["sudo", "chmod", "go+r", "/etc/apt/keyrings/githubcli-archive-keyring.gpg"],
                ["sudo", "bash", "-c",
                 "echo \"deb [arch=$(dpkg --print-architecture)"
                 " signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg]"
                 " https://cli.github.com/packages stable main\""
                 " | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null"],
                ["sudo", "apt-get", "update", "-qq"],
                ["sudo", "apt-get", "install", "-y", "--no-install-recommends", "gh"],
            ]
            for cmd in cmds:
                if run(*cmd).returncode != 0:
                    return False
            return True
        if shutil.which("dnf") and shutil.which("sudo"):
            info("Attempting dnf-based gh install …")
            cmds = [
                ["sudo", "dnf", "config-manager", "--add-repo",
                 "https://cli.github.com/packages/rpm/gh-cli.repo"],
                ["sudo", "dnf", "install", "-y", "gh"],
            ]
            for cmd in cmds:
                if run(*cmd).returncode != 0:
                    return False
            return True
    return False


def _print_gh_install_instructions() -> None:
    system = platform.system()
    print("\n  GitHub CLI (gh) is not installed. Install it using one of the methods below,\n"
          "  then re-run this script.\n")
    if system == "Windows":
        print("    WinGet (recommended):")
        print("      winget install --id GitHub.cli\n")
        print("    Manual installer: https://cli.github.com/\n")
    elif system == "Darwin":
        print("    Homebrew:")
        print("      brew install gh\n")
    else:
        print("    Debian/Ubuntu: https://github.com/cli/cli/blob/trunk/docs/install_linux.md\n")
        print("    Fedora/RHEL:")
        print("      sudo dnf config-manager --add-repo https://cli.github.com/packages/rpm/gh-cli.repo")
        print("      sudo dnf install gh\n")
    print("  Docs: https://cli.github.com/\n")


def check_gh() -> None:
    version = _gh_version()
    if version:
        ok(f"GitHub CLI  {version}")
        return

    warn("GitHub CLI (gh) not found in PATH.")
    info("Attempting automatic install …")
    if _try_install_gh():
        version = _gh_version()
        if version:
            ok(f"GitHub CLI installed: {version}")
            return
        warn("Install command succeeded but 'gh' still not in PATH. "
             "You may need to restart your shell.")

    _print_gh_install_instructions()
    sys.exit(1)


# ──────────────────────────────────────────────
# copilot CLI
# ──────────────────────────────────────────────

def _copilot_version() -> Optional[str]:
    if not shutil.which("copilot"):
        return None
    try:
        r = run("copilot", "version", capture_output=True, text=True, timeout=15)
        line = (r.stdout or r.stderr or "").splitlines()
        return line[0].strip() if line else "unknown"
    except Exception:
        return None


def _try_install_copilot() -> bool:
    system = platform.system()
    if system == "Windows":
        if shutil.which("winget"):
            info("Attempting: winget install GitHub.Copilot --silent …")
            r = run("winget", "install", "GitHub.Copilot", "--silent",
                    "--accept-package-agreements", "--accept-source-agreements")
            return r.returncode == 0
    elif system in ("Darwin", "Linux"):
        if shutil.which("brew"):
            info("Attempting: brew install copilot-cli …")
            r = run("brew", "install", "copilot-cli")
            return r.returncode == 0
    return False


def _print_copilot_install_instructions() -> None:
    system = platform.system()
    print("\n  copilot CLI is not installed. Install it using one of the methods below,\n"
          "  then re-run this script.\n")
    if system == "Windows":
        print("    WinGet:")
        print("      winget install GitHub.Copilot\n")
    else:
        print("    Homebrew:")
        print("      brew install copilot-cli\n")
    print("  Docs: https://docs.github.com/en/copilot/how-tos/copilot-cli/cli-getting-started\n")


def check_copilot() -> None:
    version = _copilot_version()
    if version:
        ok(f"copilot CLI {version}")
        return

    warn("copilot CLI not found in PATH.")
    info("Attempting automatic install …")
    if _try_install_copilot():
        version = _copilot_version()
        if version:
            ok(f"copilot CLI installed: {version}")
            return
        warn("Install command succeeded but 'copilot' still not in PATH. "
             "You may need to restart your shell.")

    _print_copilot_install_instructions()
    sys.exit(1)


# ──────────────────────────────────────────────
# Authentication
# ──────────────────────────────────────────────

def _token_in_env() -> Optional[str]:
    for var in ("COPILOT_GITHUB_TOKEN", "GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(var):
            return var
    return None


def _token_stored() -> bool:
    cfg = Path.home() / ".copilot" / "config.json"
    if not cfg.exists():
        return False
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return bool(data.get("token") or data.get("githubToken") or data.get("auth"))
    except Exception:
        return cfg.stat().st_size > 64


def _print_token_hint() -> None:
    print("\n  Alternatively, set a Personal Access Token with 'copilot' scope:\n")
    print("    Linux / macOS – add to ~/.bashrc or ~/.zshrc:")
    print("      export COPILOT_GITHUB_TOKEN=ghp_<your-token>\n")
    print("    Windows PowerShell – add to $PROFILE:")
    print("      $env:COPILOT_GITHUB_TOKEN = 'ghp_<your-token>'\n")
    print("  Token scopes needed: read:user, copilot\n")
    print("  Create one at: https://github.com/settings/tokens\n")


def check_auth() -> None:
    var = _token_in_env()
    if var:
        ok(f"Auth via ${var}")
        return

    if _token_stored():
        ok("Auth: stored credentials found (~/.copilot/config.json)")
        return

    warn("No authentication found.")
    info("Starting 'copilot login' – follow the prompts in your browser.\n")
    try:
        result = run("copilot", "login")
        if result.returncode != 0:
            warn("'copilot login' exited with an error.")
            _print_token_hint()
            sys.exit(1)
        ok("Authentication complete.")
    except KeyboardInterrupt:
        print()
        warn("Login cancelled.")
        _print_token_hint()
        sys.exit(1)
    except FileNotFoundError:
        fail("'copilot' command not found – install it first (step above).")


# ──────────────────────────────────────────────
# Python deps
# ──────────────────────────────────────────────

def install_python_deps() -> None:
    if not REQUIREMENTS.exists():
        info("No requirements.txt found in repo root – skipping.")
        return
    info(f"Installing Python dependencies from {REQUIREMENTS.name} …")
    result = run(sys.executable, "-m", "pip", "install", "-q", "-r", str(REQUIREMENTS))
    if result.returncode != 0:
        fail("pip install failed. Check the error above.")
    ok("Python dependencies installed.")


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────

def main() -> None:
    print("\n=== OpenVINO agent scripts – Environment Setup ===\n")

    check_python()
    check_gh()
    check_copilot()
    check_auth()
    install_python_deps()

    print(
        "\n  All checks passed. You can now run agent scripts.\n"
        "  Example:\n"
        "    python .github/scripts/meat/enable_operator.py my_op.md\n"
        "\n"
        "  See .github/scripts/meat/README.md for full usage.\n"
    )


if __name__ == "__main__":
    main()
