# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for meat-scripts test suite."""

import json
import os
import pathlib
import sys
import textwrap

import pytest

SCRIPTS_DIR = pathlib.Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def run_script(script: pathlib.Path, cwd, *args, extra_env=None):
    """Run a Python script as a subprocess, return CompletedProcess."""
    import subprocess

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
    )


def write_json(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Cross-platform fake-command factory
#
# Creates a callable named `name` in `bin_dir` that runs a small Python
# script.  On Windows a .bat wrapper is written; on POSIX a shell script.
# ---------------------------------------------------------------------------

def _make_stub_impl(bin_dir: pathlib.Path, name: str, body: str) -> pathlib.Path:
    """Write the Python implementation and a platform launcher."""
    impl = bin_dir / f"_stub_{name}.py"
    impl.write_text(body, encoding="utf-8")

    if sys.platform == "win32":
        launcher = bin_dir / f"{name}.bat"
        launcher.write_text(
            f'@"{sys.executable}" "{impl}" %*\r\n',
            encoding="utf-8",
        )
    else:
        launcher = bin_dir / name
        launcher.write_text(
            f'#!/bin/sh\nexec python3 "{impl}" "$@"\n',
            encoding="utf-8",
        )
        launcher.chmod(0o755)

    return launcher


def make_fake_cmd(bin_dir: pathlib.Path, name: str, *,
                  exit_code: int = 0,
                  create_ir: bool = False,
                  stdout_text: str = "") -> None:
    """
    Create a fake ``name`` command in ``bin_dir``.

    Parameters
    ----------
    exit_code : int
        Return code the stub will produce.
    create_ir : bool
        If True, the stub creates .xml/.bin files in the directory
        given by the ``--output <dir>`` argument.
    stdout_text : str
        Text to print to stdout before exiting.
    """
    body = textwrap.dedent(f"""\
        import sys, pathlib
        if {stdout_text!r}:
            print({stdout_text!r})
        args = sys.argv
        if {create_ir!r} and "--output" in args:
            out_idx = args.index("--output") + 1
            out = pathlib.Path(args[out_idx])
            out.mkdir(parents=True, exist_ok=True)
            (out / "openvino_model.xml").write_text("<xml/>")
            (out / "openvino_model.bin").write_bytes(b"")
        sys.exit({exit_code})
    """)
    _make_stub_impl(bin_dir, name, body)


def patched_env(bin_dir: pathlib.Path, base: dict | None = None) -> dict:
    """Return an env dict with *bin_dir* prepended to PATH."""
    env = (base or os.environ).copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
    return env
