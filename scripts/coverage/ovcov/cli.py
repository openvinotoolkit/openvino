# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Callable
import importlib
import os
import sys
from pathlib import Path

from .context import CoverageContext
from .models import SUPPORTED_PROFILES
from .runner import error, log


STEP_MODULES: dict[str, str] = {
    "install-deps": "ovcov.steps.install_deps",
    "configure": "ovcov.steps.configure",
    "build-install": "ovcov.steps.build_install",
    "run-cpp-tests": "ovcov.steps.run_cpp_tests",
    "run-python-tests": "ovcov.steps.run_python_tests",
    "run-js-tests": "ovcov.steps.run_js_tests",
    "collect-cpp-coverage": "ovcov.steps.collect_cpp_coverage",
    "write-summary": "ovcov.steps.write_summary",
    "package-artifacts": "ovcov.steps.package_artifacts",
}

RUN_ALL_ORDER = [
    "install-deps",
    "configure",
    "build-install",
    "run-cpp-tests",
    "run-python-tests",
    "run-js-tests",
    "collect-cpp-coverage",
    "write-summary",
    "package-artifacts",
]


def _apply_common_env(args: argparse.Namespace) -> None:
    profile = getattr(args, "profile", None)
    if profile:
        os.environ["TEST_PROFILE"] = profile

    workspace = getattr(args, "workspace", None)
    if workspace:
        os.environ["OV_WORKSPACE"] = str(Path(workspace).resolve())

    build_type = getattr(args, "build_type", None)
    if build_type:
        os.environ["CMAKE_BUILD_TYPE"] = build_type

    parallel_jobs = getattr(args, "parallel_jobs", None)
    if parallel_jobs is not None:
        os.environ["PARALLEL_JOBS"] = str(parallel_jobs)

    pytest_workers = getattr(args, "pytest_workers", None)
    if pytest_workers is not None:
        os.environ["PYTEST_XDIST_WORKERS"] = str(pytest_workers)

    js_test_concurrency = getattr(args, "js_test_concurrency", None)
    if js_test_concurrency is not None:
        os.environ["JS_TEST_CONCURRENCY"] = str(js_test_concurrency)


def _load_context(args: argparse.Namespace) -> CoverageContext:
    _apply_common_env(args)
    return CoverageContext.from_env()


def _resolve_step_handler(step_name: str) -> Callable[[CoverageContext], None]:
    module_name = STEP_MODULES.get(step_name)
    if not module_name:
        raise KeyError(f"Unknown step '{step_name}'")
    module = importlib.import_module(module_name)
    handler = getattr(module, "run", None)
    if not callable(handler):
        raise RuntimeError(f"Step module '{module_name}' does not define callable run(ctx)")
    return handler  # type: ignore[return-value]


def _run_step(ctx: CoverageContext, step_name: str) -> None:
    log(f"Starting step: {step_name}")
    handler = _resolve_step_handler(step_name)
    handler(ctx)


def _command_step(args: argparse.Namespace) -> int:
    ctx = _load_context(args)
    _run_step(ctx, args.step_name)
    return 0


def _command_run_all(args: argparse.Namespace) -> int:
    ctx = _load_context(args)

    try:
        start_index = RUN_ALL_ORDER.index(args.from_step) if args.from_step else 0
    except ValueError as exc:
        raise ValueError(f"Unknown --from step: {args.from_step}") from exc

    try:
        end_index = RUN_ALL_ORDER.index(args.to_step) if args.to_step else len(RUN_ALL_ORDER) - 1
    except ValueError as exc:
        raise ValueError(f"Unknown --to step: {args.to_step}") from exc

    if start_index > end_index:
        raise ValueError("Invalid range: --from is after --to")

    failed_steps: list[str] = []

    for step_name in RUN_ALL_ORDER[start_index : end_index + 1]:
        if step_name == "install-deps" and not args.install_deps:
            log("Skipping install-deps step (use --install-deps to enable)")
            continue

        try:
            _run_step(ctx, step_name)
        except Exception as exc:  # noqa: BLE001
            error(f"Step failed: {step_name}: {exc}")
            failed_steps.append(step_name)
            if args.strict:
                break

    log("Local coverage outputs:")
    log(f"  {ctx.workspace / 'coverage.info'}")
    log(f"  {ctx.workspace / 'python-coverage.xml'}")
    log(f"  {ctx.workspace / 'js-lcov.info'}")
    log(f"  {ctx.workspace / 'coverage-report' / 'index.html'}")
    log(f"  {ctx.io.summary_file}")

    if failed_steps:
        error(f"Completed with failed steps: {', '.join(failed_steps)}")
        return 1

    log("Completed successfully")
    return 0


def _command_list_tests(args: argparse.Namespace) -> int:
    from .config import load_cpp_tests, load_js_tests, load_python_tests

    _apply_common_env(args)
    workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or Path.cwd()).resolve()
    config_dir = workspace / "scripts" / "coverage" / "config"

    if args.suite == "cpp":
        tests = load_cpp_tests(config_dir / "tests_cpp.yml", args.profile)
        print("name\tenabled\tskip_reason\tbinary\tmode\targs\textra_env")
        for t in tests:
            print(
                "\t".join(
                    [
                        t.name,
                        "1" if t.enabled else "0",
                        t.skip_reason,
                        t.binary,
                        t.mode,
                        t.args,
                        t.extra_env,
                    ]
                )
            )
    elif args.suite == "python":
        tests = load_python_tests(config_dir / "tests_python.yml", args.profile)
        print("name\tenabled\tskip_reason\tkind\ttarget\targs\tenv\tcommand")
        for t in tests:
            print(
                "\t".join(
                    [
                        t.name,
                        "1" if t.enabled else "0",
                        t.skip_reason,
                        t.kind,
                        t.target,
                        t.args,
                        t.env,
                        t.command,
                    ]
                )
            )
    else:
        tests = load_js_tests(config_dir / "tests_js.yml", args.profile)
        print("name\tenabled\tskip_reason\tkind\tcommand")
        for t in tests:
            print(
                "\t".join(
                    [
                        t.name,
                        "1" if t.enabled else "0",
                        t.skip_reason,
                        t.kind,
                        t.command,
                    ]
                )
            )

    return 0


def _command_validate_config(args: argparse.Namespace) -> int:
    from .config import validate_configs

    _apply_common_env(args)
    workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or Path.cwd()).resolve()
    issues = validate_configs(workspace / "scripts" / "coverage" / "config")

    if issues:
        for issue in issues:
            print(f"[config][{issue.suite}] {issue.test_name}: {issue.message}")
        return 1

    print("Coverage config validation passed")
    return 0


def _add_common_options(parser: argparse.ArgumentParser, *, include_profile: bool = True) -> None:
    if include_profile:
        parser.add_argument("--profile", choices=sorted(SUPPORTED_PROFILES), default=os.environ.get("TEST_PROFILE", "cpu"))
    parser.add_argument("--workspace", default=os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE"))
    parser.add_argument("--build-type", default=os.environ.get("CMAKE_BUILD_TYPE", "Release"))
    parser.add_argument("--parallel-jobs", type=int, default=None)
    parser.add_argument("--pytest-workers", type=int, default=None)
    parser.add_argument("--js-test-concurrency", type=int, default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenVINO coverage workflow orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_all_parser = subparsers.add_parser("run-all", help="Run all coverage steps")
    _add_common_options(run_all_parser)
    run_all_parser.add_argument("--install-deps", action="store_true", help="Run install-deps step")
    run_all_parser.add_argument("--from", dest="from_step", choices=RUN_ALL_ORDER)
    run_all_parser.add_argument("--to", dest="to_step", choices=RUN_ALL_ORDER)
    run_all_parser.add_argument("--strict", action="store_true", help="Stop on first failure")
    run_all_parser.set_defaults(func=_command_run_all)

    step_parser = subparsers.add_parser("step", help="Run one coverage step")
    _add_common_options(step_parser)
    step_parser.add_argument("step_name", choices=sorted(STEP_MODULES.keys()))
    step_parser.set_defaults(func=_command_step)

    list_parser = subparsers.add_parser("list-tests", help="List resolved tests for suite/profile")
    list_parser.add_argument("--suite", required=True, choices=["cpp", "python", "js"])
    list_parser.add_argument("--profile", choices=sorted(SUPPORTED_PROFILES), default=os.environ.get("TEST_PROFILE", "cpu"))
    list_parser.add_argument("--workspace", default=os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE"))
    list_parser.set_defaults(func=_command_list_tests)

    validate_parser = subparsers.add_parser("validate-config", help="Validate YAML coverage test configs")
    validate_parser.add_argument("--workspace", default=os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE"))
    validate_parser.set_defaults(func=_command_validate_config)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
