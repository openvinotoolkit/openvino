#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Reads pytorch_models.toml and creates/updates one venv per [envs.<name>] entry: a python -m venv, an
# OpenVINO wheel install, then that env's ordered list of pip install steps (each step's requirement
# files/pip args come straight from the manifest). Uses the same cp<major><minor> wheel-glob logic
# .github/actions/install_ov_wheels/action.yml uses (excluding free-threaded "...cp311t..." wheels), so
# results are directly comparable to real CI. A venv is skipped and reused if its stamp file matches a
# hash over its manifest steps, the base interpreter, and the OpenVINO source (for wheels:<dir>, the
# selected wheel files' contents; `nightly` is not content-addressed, use --force to pick up a newer one).
#
# Usage:
#   build_envs.py --venvs-dir DIR [--manifest pytorch_models.toml] [--ov-source nightly|wheels:<dir>]
#                 [--python PATH] [--env NAME ...] [--force]
#
#   --ov-source nightly       installs --pre openvino/openvino-tokenizers from the nightly wheel index.
#   --ov-source wheels:<dir>  globs <dir> for cp<major><minor> wheels the same way CI's
#                              install_ov_wheels action does.
#   --env NAME (repeatable)   build only these envs instead of every env in the manifest.
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - CI's Python is 3.11
    import tomli as tomllib

OV_NIGHTLY_INDEX = "https://storage.openvinotoolkit.org/simple/wheels/nightly"
OV_WHEEL_NAMES = ["openvino", "openvino_tokenizers"]


def load_manifest(manifest_path: Path):
    with open(manifest_path, "rb") as f:
        return tomllib.load(f)


def resolve_step_files(manifest_dir: Path, step):
    """A step is either a plain list of requirement-file paths, or a {files, pip_args} table (for a
    step that needs extra pip flags, e.g. --no-build-isolation for llm.txt)."""
    if isinstance(step, dict):
        files = step["files"]
        pip_args = step.get("pip_args", [])
    else:
        files = step
        pip_args = []
    return [str((manifest_dir / f).resolve()) for f in files], pip_args


def interpreter_info(python_exe: str):
    """(identity string, cp<major><minor> tag) of the interpreter a venv would be created from."""
    out = subprocess.run(
        [python_exe, "-c",
         "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}'); "
         "print(sys.executable); print(sys.version)"],
        capture_output=True, text=True, check=True,
    ).stdout
    py_version = out.split("\n", 1)[0].strip()
    return out, py_version


def resolve_ov_wheels(ov_source: str, py_version: str):
    """Wheel paths to install for a `wheels:<dir>` source, or [] for `nightly`."""
    if ov_source == "nightly":
        return []
    if ov_source.startswith("wheels:"):
        wheels_dir = Path(ov_source[len("wheels:"):])
        return [find_ov_wheel(wheels_dir, n, py_version) for n in OV_WHEEL_NAMES]
    raise ValueError(f"Unknown --ov-source {ov_source!r} (expected 'nightly' or 'wheels:<dir>')")


def stamp_content(env_steps, ov_source, interpreter_id, ov_wheels):
    """Hash of every requirement file's content + the pip_args + the base interpreter + the OpenVINO
    source (including the selected wheels' contents), so a venv is rebuilt exactly when something
    that would change its installed packages changes."""
    h = hashlib.sha256()
    h.update(ov_source.encode())
    h.update(interpreter_id.encode())
    for wheel in ov_wheels:
        h.update(Path(wheel).name.encode())
        with open(wheel, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    for files, pip_args in env_steps:
        h.update("|".join(pip_args).encode())
        for f in files:
            h.update(f.encode())
            h.update(Path(f).read_bytes())
    return h.hexdigest()


def find_ov_wheel(wheels_dir: Path, name: str, py_version: str):
    # Same logic as .github/actions/install_ov_wheels/action.yml's Linux/macOS branch: prefer a
    # wheel tagged for this interpreter's cp<major><minor>, excluding free-threaded "...cp311t...".
    candidates = sorted(wheels_dir.glob(f"{name}-*cp{py_version}*.whl"))
    candidates = [c for c in candidates if f"cp{py_version}t" not in c.name]
    if candidates:
        return candidates[0]
    fallback = sorted(wheels_dir.glob(f"{name}-*.whl"))
    if not fallback:
        raise FileNotFoundError(f"No wheel found for {name} in {wheels_dir}")
    return fallback[0]


def run(cmd, log_file):
    print(f"+ {' '.join(cmd)}", file=log_file, flush=True)
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_file.write(result.stdout)
    log_file.flush()
    if result.returncode != 0:
        print(result.stdout[-4000:])
        raise RuntimeError(f"command failed (exit {result.returncode}): {' '.join(cmd)}")


def build_env(name, manifest_dir, steps_raw, venvs_dir: Path, python_exe: str, ov_source: str, force: bool):
    venv_path = venvs_dir / name
    stamp_path = venv_path / ".env_stamp"
    env_steps = [resolve_step_files(manifest_dir, s) for s in steps_raw]
    interpreter_id, py_version = interpreter_info(python_exe)
    ov_wheels = resolve_ov_wheels(ov_source, py_version)
    stamp = stamp_content(env_steps, ov_source, interpreter_id, ov_wheels)

    if not force and stamp_path.exists() and stamp_path.read_text().strip() == stamp:
        print(f"[build_envs] {name}: up to date, reusing {venv_path}")
        return venv_path

    print(f"[build_envs] {name}: building {venv_path}")
    logs_dir = venvs_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{name}.log"

    with open(log_path, "w") as log_file:
        subprocess.run(["rm", "-rf", str(venv_path)], check=True)
        run([python_exe, "-m", "venv", str(venv_path)], log_file)
        py = str(venv_path / "bin" / "python")
        run([py, "-m", "pip", "install", "--upgrade", "pip", "-q"], log_file)

        if ov_wheels:
            run([py, "-m", "pip", "install", *map(str, ov_wheels)], log_file)
        else:
            run([py, "-m", "pip", "install", "--pre", "--extra-index-url", OV_NIGHTLY_INDEX, *OV_WHEEL_NAMES], log_file)

        for files, pip_args in env_steps:
            file_flags = sum([["-r", f] for f in files], [])
            run([py, "-m", "pip", "install", *pip_args, *file_flags], log_file)

        run([py, "-m", "pip", "check"], log_file)
        freeze = subprocess.run([py, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
        (venvs_dir / "logs" / f"{name}.freeze.txt").write_text(freeze)

    stamp_path.write_text(stamp)
    print(f"[build_envs] {name}: done ({venv_path})")
    return venv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--venvs-dir", required=True)
    parser.add_argument("--manifest", default=str(Path(__file__).parent / "pytorch_models.toml"))
    parser.add_argument("--ov-source", default="nightly")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--env", action="append", default=None, help="build only these envs (repeatable)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = load_manifest(manifest_path)
    venvs_dir = Path(args.venvs_dir).resolve()
    venvs_dir.mkdir(parents=True, exist_ok=True)

    env_names = args.env if args.env else list(manifest["envs"].keys())
    built = {}
    for name in env_names:
        if name not in manifest["envs"]:
            raise KeyError(f"env {name!r} not found in manifest (have: {list(manifest['envs'].keys())})")
        steps_raw = manifest["envs"][name]["steps"]
        built[name] = str(build_env(name, manifest_dir, steps_raw, venvs_dir, args.python, args.ov_source, args.force))

    print(json.dumps(built, indent=2))


if __name__ == "__main__":
    main()
