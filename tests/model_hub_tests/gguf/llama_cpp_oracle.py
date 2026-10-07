# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Build the pinned llama.cpp CPU encoder oracle for the mmproj model tests.

The reference values come from llama.cpp itself at test time, so no stored expectations go
stale. The build fetches one upstream commit, builds only libmtmd and its dependencies, and
compiles mmproj_oracle.cpp from the GGUF frontend tests against it. Results are cached per
revision under GGUF_LLAMA_CPP_CACHE (default: ~/.cache/openvino_gguf_llama_cpp).
Set GGUF_MMPROJ_ORACLE to an already built mmproj_oracle to skip the build; the checkout is
still fetched for its gguf-py package.
"""

import contextlib
import fcntl
import os
import shutil
import subprocess
from pathlib import Path

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
# The revision the GGUF frontend's mmproj fixtures are generated with.
LLAMA_CPP_REVISION = "03fa73cb27f5c251b9528489b18d303b1366aca4"


def frontend_test_file(name: str) -> Path:
    """A file shared with src/frontends/gguf/tests, installed next to this module."""
    here = Path(__file__).resolve().parent
    for candidate in (here / name, here.parents[2] / "src" / "frontends" / "gguf" / "tests" / name):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} is neither installed next to {here} nor in the source tree")


def _run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def _compilers():
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("cc")
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("c++")
    if not cc or not cxx:
        raise RuntimeError("Building the llama.cpp oracle needs a C and C++ compiler")
    return cc, cxx


def _root() -> Path:
    cache = Path(os.environ.get("GGUF_LLAMA_CPP_CACHE", Path.home() / ".cache" / "openvino_gguf_llama_cpp"))
    return cache / LLAMA_CPP_REVISION


@contextlib.contextmanager
def _locked():
    """Serialize fetching and building across processes that share the cache."""
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield root
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _checkout(root: Path) -> Path:
    source = root / "src"
    if not (source / "CMakeLists.txt").exists():
        shutil.rmtree(source, ignore_errors=True)
        source.mkdir(parents=True)
        _run(["git", "init", "-q"], cwd=source)
        _run(["git", "fetch", "-q", "--depth", "1", LLAMA_CPP_REPO, LLAMA_CPP_REVISION], cwd=source)
        _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=source)
    return source


def llama_cpp_checkout() -> Path:
    """The pinned llama.cpp source tree, fetched on first use."""
    with _locked() as root:
        return _checkout(root)


def build_mmproj_oracle() -> Path:
    if prebuilt := os.environ.get("GGUF_MMPROJ_ORACLE"):
        return Path(prebuilt)
    with _locked() as root:
        oracle = root / "mmproj_oracle"
        if not oracle.exists():
            _build(root, _checkout(root), oracle)
        return oracle


def _build(root: Path, source: Path, oracle: Path):
    build = root / "build"
    cc, cxx = _compilers()
    _run(["cmake", "-S", source, "-B", build, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
          f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}", "-DGGML_NATIVE=OFF",
          "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
          "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_TOOLS=ON"], stdout=subprocess.DEVNULL)
    _run(["cmake", "--build", build, "--target", "mtmd"], stdout=subprocess.DEVNULL)
    libraries = build / "bin"
    _run([cxx, "-std=c++17", "-O2", frontend_test_file("mmproj_oracle.cpp"),
          "-I", source / "tools" / "mtmd", "-I", source / "include", "-I", source / "ggml" / "include",
          "-L", libraries, f"-Wl,-rpath,{libraries}",
          "-lmtmd", "-lllama", "-lggml", "-lggml-base", "-o", oracle.with_suffix(f".{os.getpid()}.tmp")])
    oracle.with_suffix(f".{os.getpid()}.tmp").replace(oracle)
