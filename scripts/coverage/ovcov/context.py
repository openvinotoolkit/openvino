# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import subprocess

from .github_io import GithubIO
from .models import Paths, ProfileFlags, SUPPORTED_PROFILES


def _repo_root(default: Path) -> Path:
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
    if profile == "cpu":
        return ProfileFlags(False, False, ("-DENABLE_INTEL_GPU=OFF", "-DENABLE_ONEDNN_FOR_GPU=OFF"), ("-DENABLE_INTEL_NPU=OFF",))
    if profile == "cpu_gpu":
        return ProfileFlags(True, False, ("-DENABLE_INTEL_GPU=ON", "-DENABLE_ONEDNN_FOR_GPU=ON"), ("-DENABLE_INTEL_NPU=OFF",))
    if profile == "cpu_npu":
        return ProfileFlags(False, True, ("-DENABLE_INTEL_GPU=OFF", "-DENABLE_ONEDNN_FOR_GPU=OFF"), ("-DENABLE_INTEL_NPU=ON",))
    if profile == "cpu_npu_gpu":
        return ProfileFlags(True, True, ("-DENABLE_INTEL_GPU=ON", "-DENABLE_ONEDNN_FOR_GPU=ON"), ("-DENABLE_INTEL_NPU=ON",))
    raise ValueError(f"Unsupported TEST_PROFILE: {profile}. Use one of: {', '.join(sorted(SUPPORTED_PROFILES))}")


@dataclass
class CoverageContext:
    workspace: Path
    build_type: str
    parallel_jobs: int
    pytest_workers: int
    js_test_concurrency: int
    test_profile: str
    cc: str
    cxx: str
    paths: Paths
    profile_flags: ProfileFlags
    io: GithubIO

    @classmethod
    def from_env(cls) -> "CoverageContext":
        workspace = Path(os.environ.get("OV_WORKSPACE") or os.environ.get("GITHUB_WORKSPACE") or str(_repo_root(Path.cwd()))).resolve()

        io = GithubIO(workspace)
        local_vars = io.load_local_env()
        for key, value in local_vars.items():
            os.environ.setdefault(key, value)

        build_type = os.environ.get("CMAKE_BUILD_TYPE", "Release")

        def _int_env(name: str, fallback: int) -> int:
            raw = os.environ.get(name)
            if raw is None or raw.strip() == "":
                return fallback
            return int(raw)

        cpu_count = os.cpu_count() or 1
        parallel_jobs = _int_env("PARALLEL_JOBS", cpu_count)
        pytest_workers = _int_env("PYTEST_XDIST_WORKERS", 1)
        js_concurrency = _int_env("JS_TEST_CONCURRENCY", 1)

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
        os.environ["PARALLEL_JOBS"] = str(parallel_jobs)
        os.environ["PYTEST_XDIST_WORKERS"] = str(pytest_workers)
        os.environ["JS_TEST_CONCURRENCY"] = str(js_concurrency)
        os.environ["TEST_PROFILE"] = test_profile
        os.environ["RUN_GPU_TESTS"] = "true" if profile_flags.run_gpu_tests else "false"
        os.environ["RUN_NPU_TESTS"] = "true" if profile_flags.run_npu_tests else "false"

        return cls(
            workspace=workspace,
            build_type=build_type,
            parallel_jobs=parallel_jobs,
            pytest_workers=pytest_workers,
            js_test_concurrency=js_concurrency,
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
    def run_npu_tests(self) -> bool:
        return self.profile_flags.run_npu_tests

    @property
    def gpu_flags(self) -> tuple[str, ...]:
        return self.profile_flags.gpu_flags

    @property
    def npu_flags(self) -> tuple[str, ...]:
        return self.profile_flags.npu_flags

    def log_profile(self) -> None:
        print(f"[coverage] TEST_PROFILE={self.test_profile}")
        print(f"[coverage] RUN_GPU_TESTS={'true' if self.run_gpu_tests else 'false'}")
        print(f"[coverage] RUN_NPU_TESTS={'true' if self.run_npu_tests else 'false'}")
        print(f"[coverage] GPU_FLAGS={' '.join(self.gpu_flags)}")
        print(f"[coverage] NPU_FLAGS={' '.join(self.npu_flags)}")
