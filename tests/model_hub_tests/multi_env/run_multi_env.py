#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# What: single CI-facing entry point that replaces job_pytorch_models_tests.yml's long sequence of
#       "install deps for group X" + "run group X" step pairs with one command per model_scope. Reads
#       pytorch_models.toml (the manifest describing every env and every pytest invocation for that
#       job), builds/reuses one venv per env via build_envs.build_env, and for the requested scope runs
#       every matching [[runs]] entry once per env in its `envs` list.
# How: for each selected (run, env) pair it launches `<venv>/bin/python -m pytest <run.paths> -m
#      <marker> <run.pytest_args...> --junitxml=... [--html=...]` as a subprocess with OV_TEST_ENV,
#      OV_TEST_DEFAULT_ENV, PYTHONPATH, TEST_DEVICE and the run's extra_env set. Per-env JUnit XML files
#      for one run are merged into a single combined <testsuite> (pytest-html reports are kept separate
#      per env instead -- merging two self-contained HTML reports cleanly isn't worth the complexity for
#      a report humans skim rather than machines parse). Every (run, env) pair must exit 0; anything
#      else is a hard failure.
# Why: a single shared venv, reinstalled in place group by group, can only ever test one
#      transformers/optimum-intel version at a time. This dispatcher lets pytorch_models.toml describe
#      an arbitrary number of independent, reproducible per-env venvs and drives all of them from one
#      CI step per model_scope, without changing what each test module asserts.
import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_envs  # noqa: E402

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "pytorch_models.toml"

# Maps the job's model_scope input to the pytest marker each scope selects -- both nightly scopes run
# the same `-m nightly` marker, they only differ in which [[runs]] entries list them in `scopes`.
SCOPE_MARKERS = {
    "precommit": "precommit",
    "nightly_scope1": "nightly",
    "nightly_scope2": "nightly",
}

EXIT_OK_CODES = {0}


@dataclass
class Invocation:
    run_name: str
    env_name: str
    cmd: list
    env_vars: dict
    junit_path: Path


@dataclass
class ExecutionResult:
    invocation: Invocation
    returncode: int


def select_runs(manifest: dict, scope: str, only_names: Optional[list] = None) -> list:
    """[[runs]] entries whose `scopes` includes the requested scope, optionally narrowed to
    --run NAME. Order is preserved from the manifest."""
    selected = []
    for run in manifest.get("runs", []):
        if scope not in run.get("scopes", []):
            continue
        if only_names and run["name"] not in only_names:
            continue
        selected.append(run)
    return selected


def build_pairs(runs: list) -> list:
    return [(run, env_name) for run in runs for env_name in run["envs"]]


def _html_path_for(run: dict, env_name: str, marker: str, results_dir: Path) -> Optional[Path]:
    html_name = run.get("html_name")
    if not html_name:
        return None
    html_file = html_name.format(scope=marker)
    if len(run["envs"]) > 1:
        stem, dot, ext = html_file.rpartition(".")
        html_file = f"{stem}.{env_name}.{ext}" if dot else f"{html_file}.{env_name}"
    return results_dir / html_file


def _resolve_extra_env_value(value: str, results_dir: Path) -> str:
    # Plain filenames (e.g. OP_REPORT_FILE) are resolved against the shared results_dir, since every
    # group appends to the same absolute-path log file; an already absolute/relative-with-separators
    # value is left untouched.
    if os.path.isabs(value) or os.sep in value:
        return value
    return str(results_dir / value)


def build_pytest_invocation(python_exe: str, run: dict, env_name: str, scope: str,
                             manifest_dir: Path, results_dir: Path) -> Invocation:
    marker = SCOPE_MARKERS[scope]
    paths = [str((manifest_dir / p).resolve()) for p in run["paths"]]
    # "TEST-torch"-prefixed and flat (no per-run subdirectory) so every result file matches the
    # existing `${INSTALL_TEST_DIR}/TEST-torch*` glob the job's "Upload Test Results" step already
    # uses -- keeps that step unchanged. A single-env run's file is already the combined result (see
    # main()'s merge step), so it skips the "_<env>" suffix a multi-env run needs to stay distinct.
    junit_filename = (f"TEST-torch_{run['name']}.junit.xml" if len(run["envs"]) == 1
                       else f"TEST-torch_{run['name']}_{env_name}.junit.xml")
    junit_path = results_dir / junit_filename

    cmd = [python_exe, "-m", "pytest", *paths]

    pytest_args = list(run.get("pytest_args", []))
    if "-m" not in pytest_args:
        cmd += ["-m", marker]
    cmd += pytest_args

    xdist_n = run.get("xdist_n")
    if xdist_n:
        cmd += ["-n", str(xdist_n)]

    cmd += [f"--junitxml={junit_path}"]

    html_path = _html_path_for(run, env_name, marker, results_dir)
    if html_path is not None:
        cmd += [f"--html={html_path}", "--self-contained-html"]

    env_vars = os.environ.copy()
    # Emulates `source <venv>/bin/activate` (VIRTUAL_ENV + venv's bin/ prepended to PATH) without
    # sourcing it: some tests shell out to a bare `python`/`pip` (e.g. a model's own custom-ops build
    # script) instead of sys.executable, and would otherwise silently pick up whatever interpreter is
    # first on PATH -- not this env's -- since launching `<venv>/bin/python` directly doesn't activate
    # the venv the way CI's own single shared, actually-activated interpreter historically did.
    # NOTE: .absolute(), not .resolve() -- venv/bin/python is itself a symlink (-> python3.X ->
    # /usr/bin/python3.X), so resolve() would dereference it down to /usr/bin, putting the *system*
    # interpreter's directory on PATH instead of the venv's own bin/.
    venv_bin_dir = str(Path(python_exe).absolute().parent)
    env_vars["VIRTUAL_ENV"] = str(Path(venv_bin_dir).parent)
    env_vars["PATH"] = venv_bin_dir + os.pathsep + env_vars.get("PATH", "")
    env_vars["OV_TEST_ENV"] = env_name
    env_vars["OV_TEST_DEFAULT_ENV"] = run["default_env"]
    env_vars["TEST_DEVICE"] = "CPU"
    model_hub_tests_dir = str(manifest_dir.parent)
    existing_pythonpath = env_vars.get("PYTHONPATH")
    env_vars["PYTHONPATH"] = (model_hub_tests_dir + os.pathsep + existing_pythonpath
                               if existing_pythonpath else model_hub_tests_dir)
    for key, value in run.get("extra_env", {}).items():
        env_vars[key] = _resolve_extra_env_value(value, results_dir)

    return Invocation(run_name=run["name"], env_name=env_name, cmd=cmd, env_vars=env_vars,
                       junit_path=junit_path)


def classify_exit_code(returncode: int) -> bool:
    return returncode in EXIT_OK_CODES


def aggregate_exit_codes(returncodes) -> int:
    return 0 if all(classify_exit_code(rc) for rc in returncodes) else 1


def _get_testsuite_element(root: ET.Element) -> ET.Element:
    if root.tag == "testsuite":
        return root
    testsuite = root.find("testsuite")
    if testsuite is None:
        raise ValueError("no <testsuite> element found in JUnit XML")
    return testsuite


_SUMMED_ATTRS = ("tests", "failures", "errors", "skipped")


def merge_junit_files(input_paths, suite_name: str):
    """Merges N per-env JUnit XML files (one <testsuite> each) into one combined <testsuite>: all
    <testcase> children concatenated, counters summed, wrapped in a <testsuites> root so any junit
    consumer sees a well-formed document either way. Returns (ElementTree, missing_paths)."""
    combined = ET.Element("testsuite", name=suite_name)
    totals = {attr: 0 for attr in _SUMMED_ATTRS}
    time_total = 0.0
    missing = []

    for path in input_paths:
        path = Path(path)
        if not path.exists():
            missing.append(str(path))
            continue
        testsuite = _get_testsuite_element(ET.parse(path).getroot())
        for testcase in list(testsuite):
            combined.append(testcase)
        for attr in _SUMMED_ATTRS:
            totals[attr] += int(testsuite.get(attr, 0) or 0)
        time_total += float(testsuite.get("time", 0) or 0)

    for attr, value in totals.items():
        combined.set(attr, str(value))
    combined.set("time", f"{time_total:.3f}")

    wrapper = ET.Element("testsuites")
    wrapper.append(combined)
    return ET.ElementTree(wrapper), missing


def _run_one(invocation: Invocation) -> ExecutionResult:
    invocation.junit_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run_multi_env] starting {invocation.run_name}/{invocation.env_name}: "
          f"{' '.join(invocation.cmd)}", flush=True)
    proc = subprocess.run(invocation.cmd, env=invocation.env_vars)
    print(f"[run_multi_env] finished {invocation.run_name}/{invocation.env_name}: "
          f"exit={proc.returncode}", flush=True)
    return ExecutionResult(invocation=invocation, returncode=proc.returncode)


def execute_invocations(invocations):
    # Sequential on purpose: running (run, env) pairs concurrently was measured on an 8-core/32GB
    # runner and rejected -- memory peaked at the configured ceiling and detectron2's runtime pip
    # install intermittently produced a corrupted package under the contention, with no net time
    # savings (each pair is itself CPU/memory-heavy: a torch import, model export/compile, or a
    # native build).
    return [_run_one(inv) for inv in invocations]


def print_plan(manifest: dict, manifest_dir: Path, args, selected_runs: list, pairs: list,
               results_dir: Path):
    marker = SCOPE_MARKERS[args.scope]
    print(f"scope={args.scope} marker={marker}")
    print(f"manifest={manifest_dir / 'pytorch_models.toml'}")
    print(f"venvs_dir={args.venvs_dir} ov_source={args.ov_source} python={args.python}")
    print(f"results_dir={results_dir}")

    unique_envs = sorted({env_name for _, env_name in pairs})
    print("\nEnvs to build/reuse:")
    for env_name in unique_envs:
        steps_raw = manifest["envs"][env_name]["steps"]
        print(f"  [{env_name}]")
        for step in steps_raw:
            files, pip_args = build_envs.resolve_step_files(manifest_dir, step)
            args_str = (" ".join(pip_args) + " ") if pip_args else ""
            print(f"    pip install {args_str}" + " ".join(f"-r {f}" for f in files))

    print("\nRuns:")
    for run in selected_runs:
        print(f"  [{run['name']}] scopes={run['scopes']} envs={run['envs']} "
              f"default_env={run['default_env']}")
        for env_name in run["envs"]:
            invocation = build_pytest_invocation("<venv>/bin/python", run, env_name, args.scope,
                                                  manifest_dir, results_dir)
            print(f"    ({env_name}) {' '.join(invocation.cmd)}")
            print(f"        OV_TEST_ENV={invocation.env_vars['OV_TEST_ENV']} "
                  f"OV_TEST_DEFAULT_ENV={invocation.env_vars['OV_TEST_DEFAULT_ENV']}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Single-command dispatcher for job_pytorch_models_tests.yml's multi-env test "
                    "suites: builds/reuses one venv per manifest env and runs pytest in each.")
    parser.add_argument("--scope", required=True, choices=sorted(SCOPE_MARKERS),
                        help="mirrors the job's model_scope input")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--venvs-dir", required=True)
    parser.add_argument("--results-dir", default=None,
                        help="default: <venvs-dir>/results")
    parser.add_argument("--ov-source", default="nightly", help="'nightly' or 'wheels:<dir>'")
    parser.add_argument("--python", default=sys.executable,
                        help="interpreter used to create each venv")
    parser.add_argument("--run", action="append", default=None, dest="run_names", metavar="NAME",
                        help="restrict to these [[runs]] names (repeatable); default: all runs "
                             "matching --scope")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the venv-build and pytest plan without building anything or "
                             "running any test")
    parser.add_argument("--force-venvs", action="store_true",
                        help="forces build_env to rebuild even if its stamp file is up to date")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    manifest = build_envs.load_manifest(manifest_path)

    selected_runs = select_runs(manifest, args.scope, args.run_names)
    if not selected_runs:
        print(f"[run_multi_env] no runs match scope={args.scope!r}"
              + (f" and --run {args.run_names}" if args.run_names else ""))
        return 0

    pairs = build_pairs(selected_runs)
    results_dir = (Path(args.results_dir) if args.results_dir
                    else Path(args.venvs_dir).resolve() / "results")

    if args.dry_run:
        print_plan(manifest, manifest_dir, args, selected_runs, pairs, results_dir)
        return 0

    venvs_dir = Path(args.venvs_dir).resolve()
    venvs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # A failed venv build (bad pin, transient network error, ...) must not take down every other
    # (run, env) pair that doesn't depend on it -- each env is built independently and a failure is
    # recorded rather than raised, so the rest of the scope still runs and reports its own results.
    built_venvs = {}
    failed_envs = {}
    for env_name in sorted({env_name for _, env_name in pairs}):
        steps_raw = manifest["envs"][env_name]["steps"]
        try:
            built_venvs[env_name] = build_envs.build_env(
                env_name, manifest_dir, steps_raw, venvs_dir, args.python, args.ov_source,
                args.force_venvs)
        except Exception as exc:  # noqa: BLE001 - reported below, not a crash
            failed_envs[env_name] = str(exc)
            print(f"[run_multi_env] ERROR: env {env_name!r} failed to build, skipping every "
                  f"(run, env) pair that needs it: {exc}")

    invocations = []
    for run, env_name in pairs:
        if env_name in failed_envs:
            continue
        python_exe = str(Path(built_venvs[env_name]) / "bin" / "python")
        invocations.append(build_pytest_invocation(python_exe, run, env_name, args.scope,
                                                     manifest_dir, results_dir))

    results = execute_invocations(invocations)

    for run in selected_runs:
        if len(run["envs"]) == 1:
            continue  # that env's own file is already the combined result, nothing to merge
        run_results = [r for r in results if r.invocation.run_name == run["name"]]
        junit_paths = [r.invocation.junit_path for r in run_results]
        combined_path = results_dir / f"TEST-torch_{run['name']}.junit.xml"
        tree, missing = merge_junit_files(junit_paths, run["name"])
        tree.write(str(combined_path), encoding="UTF-8", xml_declaration=True)
        if missing:
            print(f"[run_multi_env] WARNING: {run['name']}: missing JUnit file(s), not merged: "
                  f"{missing}")

    print("\n[run_multi_env] summary:")
    for env_name, error in failed_envs.items():
        print(f"  env {env_name}: BUILD FAILED ({error})")
    for result in results:
        ok = classify_exit_code(result.returncode)
        print(f"  {result.invocation.run_name}/{result.invocation.env_name}: "
              f"exit={result.returncode} ({'OK' if ok else 'FAIL'})")

    if failed_envs:
        return 1
    return aggregate_exit_codes([r.returncode for r in results])


if __name__ == "__main__":
    sys.exit(main())
