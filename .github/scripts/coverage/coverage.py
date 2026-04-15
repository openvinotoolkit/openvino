# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
import logging
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = SCRIPT_DIR / "config"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
# Stable import alias used by step modules.
sys.modules.setdefault("coverage_workflow", sys.modules[__name__])


SUPPORTED_PROFILES = {"cpu", "gpu"}


def _build_logger() -> logging.Logger:
    """Create the shared logger used by the coverage helpers."""
    logger = logging.getLogger("coverage")
    if logger.handlers:
        return logger

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[coverage][%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


LOGGER = _build_logger()


@dataclass(frozen=True)
class CppTestCase:
    name: str
    enabled: bool
    skip_reason: str
    binary: str
    mode: str
    args: str
    extra_env: str


@dataclass(frozen=True)
class PythonTestCase:
    name: str
    enabled: bool
    skip_reason: str
    kind: str
    target: str
    args: str
    env: str
    command: str


@dataclass(frozen=True)
class JsTestCase:
    name: str
    enabled: bool
    skip_reason: str
    kind: str
    command: str


@dataclass(frozen=True)
class Paths:
    workspace: Path
    build_dir: Path
    build_js_dir: Path
    install_pkg_dir: Path
    bin_dir: Path
    js_dir: Path
    model_path: Path


@dataclass(frozen=True)
class ProfileFlags:
    run_gpu_tests: bool
    gpu_flags: tuple[str, ...]


@dataclass(frozen=True)
class ConfigValidationIssue:
    suite: str
    test_name: str
    message: str


class GithubIO:
    """Manage GitHub Actions env and summary files with a local fallback."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._github_env = os.environ.get("GITHUB_ENV", "").strip()
        self._github_summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()
        self._summary_enabled = os.environ.get("COVERAGE_WRITE_STEP_SUMMARY", "true").strip().lower() != "false"

        self.local_state_dir = workspace / ".tmp" / "coverage-local"
        self.local_env_file = self.local_state_dir / "github_env"
        self.local_summary_file = self.local_state_dir / "step_summary.md"

        self.local_state_dir.mkdir(parents=True, exist_ok=True)
        self.local_env_file.touch(exist_ok=True)
        self.local_summary_file.touch(exist_ok=True)

    @property
    def env_file(self) -> Path:
        if self._github_env:
            return Path(self._github_env)
        return self.local_env_file

    @property
    def summary_file(self) -> Path:
        if self._github_summary:
            return Path(self._github_summary)
        return self.local_summary_file

    @property
    def is_github_mode(self) -> bool:
        return bool(self._github_env)

    def load_local_env(self) -> dict[str, str]:
        """Load locally persisted workflow variables when not in Actions."""
        if self.is_github_mode:
            return {}

        loaded: dict[str, str] = {}
        if not self.local_env_file.exists():
            return loaded

        for raw_line in self.local_env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            loaded[key] = value
        return loaded

    def export_env(self, key: str, value: str) -> None:
        """Append an environment variable to the active env file."""
        with self.env_file.open("a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
        os.environ[key] = value

    def append_summary(self, text: str) -> None:
        """Append text to the step summary when summary output is enabled."""
        if not self._summary_enabled:
            return
        with self.summary_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")


def run_cmd(
    cmd: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    shell: bool = False,
) -> int:
    """Run a command and optionally fail when it returns non-zero."""
    display = cmd if isinstance(cmd, str) else " ".join(shlex.quote(part) for part in cmd)
    LOGGER.info("$ %s", display)
    completed = subprocess.run(cmd, cwd=cwd, env=env, shell=shell, text=True)
    if check and completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {display}")
    return completed.returncode


def run_cmd_capture(cmd: Sequence[str], *, cwd: Path | None = None, check: bool = True) -> str:
    """Run a command and return captured stdout."""
    display = " ".join(shlex.quote(part) for part in cmd)
    LOGGER.info("$ %s", display)
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if check and completed.returncode != 0:
        stderr = completed.stderr.strip()
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {display}\n{stderr}")
    return completed.stdout


def env_from_assignments(assignments: str | None, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Apply shell-style KEY=VALUE assignments to an environment mapping."""
    env = dict(base_env or os.environ)
    if not assignments:
        return env

    for token in shlex.split(assignments):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        env[key] = value
    return env


def _repo_root(default: Path) -> Path:
    """Return the git repository root when available."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if output:
            return Path(output)
    except Exception:
        pass
    return default


def _profile_flags(profile: str) -> ProfileFlags:
    """Resolve build flags and runtime switches for a test profile."""
    if profile == "cpu":
        return ProfileFlags(False, ("-DENABLE_INTEL_GPU=OFF", "-DENABLE_ONEDNN_FOR_GPU=OFF"))
    if profile == "gpu":
        return ProfileFlags(True, ("-DENABLE_INTEL_GPU=ON", "-DENABLE_ONEDNN_FOR_GPU=ON"))
    raise ValueError(f"Unsupported TEST_PROFILE: {profile}. Use one of: {', '.join(sorted(SUPPORTED_PROFILES))}")


@dataclass
class CoverageContext:
    """Runtime configuration shared by coverage workflow steps."""

    workspace: Path
    build_type: str
    branch_coverage: bool
    test_profile: str
    cc: str
    cxx: str
    paths: Paths
    profile_flags: ProfileFlags
    io: GithubIO

    @classmethod
    def from_env(cls) -> "CoverageContext":
        """Build a coverage context from environment variables."""
        workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or str(_repo_root(Path.cwd()))).resolve()

        io = GithubIO(workspace)
        local_vars = io.load_local_env()
        for key, value in local_vars.items():
            os.environ.setdefault(key, value)

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")

        def _bool_env(name: str, fallback: bool) -> bool:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return fallback
            return raw.strip().lower() in {"1", "true", "yes", "on"}

        branch_coverage = _bool_env("ENABLE_BRANCH_COVERAGE", False)

        test_profile = os.environ.get("TEST_PROFILE", "cpu").strip()
        if test_profile not in SUPPORTED_PROFILES:
            raise ValueError(f"Unsupported TEST_PROFILE: {test_profile}. Use one of: {', '.join(sorted(SUPPORTED_PROFILES))}")

        cc = os.environ.get("CC", "gcc")
        cxx = os.environ.get("CXX", "g++")

        paths = Paths(
            workspace=workspace,
            build_dir=Path(os.environ.get("BUILD_DIR", str(workspace / "build"))),
            build_js_dir=Path(os.environ.get("BUILD_JS_DIR", str(workspace / "build_js"))),
            install_pkg_dir=Path(os.environ.get("INSTALL_PKG_DIR", str(workspace / "install_pkg"))),
            bin_dir=Path(os.environ.get("BIN_DIR", str(workspace / "bin" / "intel64" / build_type))),
            js_dir=Path(os.environ.get("JS_DIR", str(workspace / "src" / "bindings" / "js" / "node"))),
            model_path=Path(os.environ.get("MODEL_PATH", str(workspace / "src" / "core" / "tests" / "models" / "ir" / "add_abc.xml"))),
        )

        profile_flags = _profile_flags(test_profile)

        os.environ["OV_WORKSPACE"] = str(workspace)
        os.environ["CMAKE_BUILD_TYPE"] = build_type
        os.environ["ENABLE_BRANCH_COVERAGE"] = "true" if branch_coverage else "false"
        os.environ["TEST_PROFILE"] = test_profile
        os.environ["RUN_GPU_TESTS"] = "true" if profile_flags.run_gpu_tests else "false"

        return cls(
            workspace=workspace,
            build_type=build_type,
            branch_coverage=branch_coverage,
            test_profile=test_profile,
            cc=cc,
            cxx=cxx,
            paths=paths,
            profile_flags=profile_flags,
            io=io,
        )

    @property
    def run_gpu_tests(self) -> bool:
        return self.profile_flags.run_gpu_tests

    @property
    def gpu_flags(self) -> tuple[str, ...]:
        return self.profile_flags.gpu_flags

    def log_profile(self) -> None:
        """Print the resolved profile and accelerator flags."""
        LOGGER.info("TEST_PROFILE=%s", self.test_profile)
        LOGGER.info("ENABLE_BRANCH_COVERAGE=%s", "true" if self.branch_coverage else "false")
        LOGGER.info("RUN_GPU_TESTS=%s", "true" if self.run_gpu_tests else "false")
        LOGGER.info("GPU_FLAGS=%s", " ".join(self.gpu_flags))


def _as_text(value: Any) -> str:
    """Normalize a value into the string form used by configs."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _resolve_profile_value(value: Any, profile: str) -> str:
    """Pick an explicit profile-specific config value when a mapping is provided."""
    if isinstance(value, dict):
        return _as_text(value.get(profile, ""))
    return _as_text(value)


def _configured_profiles(test: dict[str, Any]) -> tuple[str, ...]:
    """Return the explicit profiles configured for a test."""
    profiles = test.get("profiles")
    if not isinstance(profiles, list):
        return ()
    return tuple(_as_text(profile).strip() for profile in profiles)


def _resolve_enabled(test: dict[str, Any]) -> tuple[bool, str]:
    """Decide whether a configured test is enabled."""
    skip_reason = _as_text(test.get("skip_reason", "")).strip()
    if skip_reason:
        return False, skip_reason

    return True, ""


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file used by the coverage tooling."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required in the coverage runtime environment.") from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML root in {path}")
    return data


def _load_tests(path: Path, suite: str) -> list[dict[str, Any]]:
    """Load and validate raw test definitions for one suite."""
    data = _load_yaml(path)
    found_suite = _as_text(data.get("suite", "")).strip()
    if found_suite != suite:
        raise ValueError(f"Invalid suite in {path}: expected '{suite}', got '{found_suite}'")
    tests = data.get("tests")
    if not isinstance(tests, list):
        raise ValueError(f"Invalid config: 'tests' list is missing in {path}")
    return tests


def load_cpp_tests(path: Path, profile: str) -> list[CppTestCase]:
    """Load resolved C++ test definitions for a profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[CppTestCase] = []
    for test in _load_tests(path, "cpp"):
        if profile not in _configured_profiles(test):
            continue
        enabled, reason = _resolve_enabled(test)
        loaded.append(
            CppTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                binary=_as_text(test.get("binary", "")).strip(),
                mode=_as_text(test.get("mode", "gtest_single")).strip() or "gtest_single",
                args=_resolve_profile_value(test.get("args", ""), profile).strip(),
                extra_env=_resolve_profile_value(test.get("extra_env", ""), profile).strip(),
            )
        )
    return loaded


def load_python_tests(path: Path, profile: str) -> list[PythonTestCase]:
    """Load resolved Python test definitions for a profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[PythonTestCase] = []
    for test in _load_tests(path, "python"):
        if profile not in _configured_profiles(test):
            continue
        enabled, reason = _resolve_enabled(test)
        loaded.append(
            PythonTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                kind=_as_text(test.get("kind", "pytest")).strip() or "pytest",
                target=_resolve_profile_value(test.get("target", ""), profile).strip(),
                args=_resolve_profile_value(test.get("args", ""), profile).strip(),
                env=_resolve_profile_value(test.get("env", ""), profile).strip(),
                command=_resolve_profile_value(test.get("command", ""), profile).strip(),
            )
        )
    return loaded


def load_js_tests(path: Path, profile: str) -> list[JsTestCase]:
    """Load resolved JS test definitions for a profile."""
    if profile not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported profile: {profile}")

    loaded: list[JsTestCase] = []
    for test in _load_tests(path, "js"):
        if profile not in _configured_profiles(test):
            continue
        enabled, reason = _resolve_enabled(test)
        loaded.append(
            JsTestCase(
                name=_as_text(test.get("name", "")).strip(),
                enabled=enabled,
                skip_reason=reason,
                kind=_as_text(test.get("kind", "command")).strip() or "command",
                command=_resolve_profile_value(test.get("command", ""), profile).strip(),
            )
        )
    return loaded


def _profile_value_fields(suite: str) -> tuple[str, ...]:
    """Return config fields that may contain per-profile mappings."""
    if suite == "cpp":
        return ("args", "extra_env")
    if suite == "python":
        return ("target", "args", "env", "command")
    if suite == "js":
        return ("command",)
    return ()


def validate_configs(config_dir: Path) -> list[ConfigValidationIssue]:
    """Validate coverage YAML configs and return any issues found."""
    issues: list[ConfigValidationIssue] = []

    suites = {
        "cpp": config_dir / "tests_cpp.yml",
        "python": config_dir / "tests_python.yml",
        "js": config_dir / "tests_js.yml",
    }

    for suite, path in suites.items():
        tests = _load_tests(path, suite)
        for idx, test in enumerate(tests):
            name = _as_text(test.get("name", f"<index:{idx}>"))
            profiles = test.get("profiles")
            expected_profiles = _configured_profiles(test)
            if not _as_text(test.get("name", "")).strip():
                issues.append(ConfigValidationIssue(suite, name, "missing 'name'"))
            if not isinstance(profiles, list) or not profiles:
                issues.append(ConfigValidationIssue(suite, name, "'profiles' must be a non-empty list"))
            else:
                for profile_name in expected_profiles:
                    if profile_name not in SUPPORTED_PROFILES:
                        issues.append(ConfigValidationIssue(suite, name, f"unsupported profile '{profile_name}'"))
            for field in _profile_value_fields(suite):
                value = test.get(field)
                if not isinstance(value, dict):
                    continue
                mapped_profiles = {_as_text(profile).strip() for profile in value}
                for profile_name in expected_profiles:
                    if profile_name in SUPPORTED_PROFILES and profile_name not in mapped_profiles:
                        issues.append(ConfigValidationIssue(suite, name, f"'{field}' missing '{profile_name}' value"))
            if suite == "cpp" and not _as_text(test.get("binary", "")).strip():
                issues.append(ConfigValidationIssue(suite, name, "missing 'binary'"))
            if suite == "python":
                kind = _as_text(test.get("kind", "pytest")).strip() or "pytest"
                if kind not in {"pytest", "pytest_if_dir", "command"}:
                    issues.append(ConfigValidationIssue(suite, name, f"unsupported kind '{kind}'"))
            if suite == "js":
                kind = _as_text(test.get("kind", "command")).strip() or "command"
                if kind != "command":
                    issues.append(ConfigValidationIssue(suite, name, f"unsupported kind '{kind}'"))

    return issues


STEP_MODULES: dict[str, str] = {
    "run-cpp-tests": "run_tests:run_cpp",
    "run-python-tests": "run_tests:run_python",
    "run-js-tests": "run_tests:run_js",
    "collect-cpp-coverage": "collect_cpp_coverage",
}


def _apply_common_env(args: argparse.Namespace) -> None:
    """Project CLI options into environment variables used by steps."""
    profile = getattr(args, "profile", None)
    if profile:
        os.environ["TEST_PROFILE"] = profile

    workspace = getattr(args, "workspace", None)
    if workspace:
        os.environ["OV_WORKSPACE"] = str(Path(workspace).resolve())

    build_type = getattr(args, "build_type", None)
    if build_type:
        os.environ["CMAKE_BUILD_TYPE"] = build_type

def _load_context(args: argparse.Namespace) -> CoverageContext:
    """Create a coverage context for the current command."""
    _apply_common_env(args)
    return CoverageContext.from_env()


def _resolve_step_handler(step_name: str) -> Callable[[CoverageContext], None]:
    """Import and return the handler for a named workflow step."""
    handler_spec = STEP_MODULES.get(step_name)
    if not handler_spec:
        raise KeyError(f"Unknown step '{step_name}'")
    module_name, _, handler_name = handler_spec.partition(":")
    module = importlib.import_module(module_name)
    handler = getattr(module, handler_name or "run", None)
    if not callable(handler):
        raise RuntimeError(f"Step handler '{handler_spec}' is not callable")
    return handler  # type: ignore[return-value]


def _run_step(ctx: CoverageContext, step_name: str) -> None:
    """Execute one coverage workflow step."""
    LOGGER.info("Starting step: %s", step_name)
    handler = _resolve_step_handler(step_name)
    handler(ctx)


def _command_step(args: argparse.Namespace) -> int:
    """Run a single named workflow step."""
    ctx = _load_context(args)
    _run_step(ctx, args.step_name)
    return 0


def _command_list_tests(args: argparse.Namespace) -> int:
    """Print resolved tests for a suite/profile pair."""
    _apply_common_env(args)
    workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or Path.cwd()).resolve()
    config_dir = CONFIG_DIR

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
    """Validate the coverage YAML configuration files."""
    _apply_common_env(args)
    workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or Path.cwd()).resolve()
    issues = validate_configs(CONFIG_DIR)

    if issues:
        for issue in issues:
            print(f"[config][{issue.suite}] {issue.test_name}: {issue.message}")
        return 1

    print("Coverage config validation passed")
    return 0


def _add_common_options(parser: argparse.ArgumentParser, *, include_profile: bool = True) -> None:
    """Attach shared CLI options used by coverage commands."""
    if include_profile:
        parser.add_argument("--profile", choices=sorted(SUPPORTED_PROFILES), default=os.environ.get("TEST_PROFILE", "cpu"))
    parser.add_argument("--workspace", default=os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE"))
    parser.add_argument("--build-type", default=os.environ.get("CMAKE_BUILD_TYPE", "Release"))


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser for coverage tooling."""
    parser = argparse.ArgumentParser(description="OpenVINO coverage test helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

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
    """Parse arguments and run the selected CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("%s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
