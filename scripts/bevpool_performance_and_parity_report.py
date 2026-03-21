#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


BENCHMARK_THROUGHPUT_RE = re.compile(r"Throughput:\s*([0-9]+(?:\.[0-9]+)?)\s*FPS", re.IGNORECASE)
BENCHMARK_LATENCY_RE = re.compile(r"Latency:\s*([0-9]+(?:\.[0-9]+)?)\s*ms", re.IGNORECASE)
GPU_REF_RE = re.compile(
    r"\[GPU vs REF\]\s+max_abs=([0-9eE+\-.]+),\s+mean_abs=([0-9eE+\-.]+),\s+max_rel=([0-9eE+\-.]+),\s+mean_rel=([0-9eE+\-.]+)"
)
CPU_REF_RE = re.compile(
    r"\[CPU vs REF\]\s+max_abs=([0-9eE+\-.]+),\s+mean_abs=([0-9eE+\-.]+),\s+max_rel=([0-9eE+\-.]+),\s+mean_rel=([0-9eE+\-.]+)"
)


@dataclass
class BenchResult:
    latency_ms: Optional[float]
    throughput_fps: Optional[float]
    raw: str


@dataclass
class ErrResult:
    max_abs: Optional[float]
    max_rel: Optional[float]
    raw: str


def prepend_path(env: Dict[str, str], key: str, value: str) -> None:
    current = env.get(key, "")
    env[key] = value if not current else f"{value}:{current}"


def run_cmd(command: list[str], cwd: Path, env: Dict[str, str]) -> str:
    proc = subprocess.run(command, cwd=str(cwd), env=env, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(command)}\n{output}")
    return output


def run_benchmark(cwd: Path, benchmark_bin: Path, model: Path, device: str, shape: str, niter: int, nireq: int, env: Dict[str, str]) -> BenchResult:
    cmd = [
        str(benchmark_bin),
        "-m",
        str(model),
        "-d",
        device,
        "-shape",
        shape,
        "-niter",
        str(niter),
        "--nireq",
        str(nireq),
    ]
    out = run_cmd(cmd, cwd, env)

    lat_match = BENCHMARK_LATENCY_RE.search(out)
    thr_match = BENCHMARK_THROUGHPUT_RE.search(out)
    return BenchResult(
        latency_ms=float(lat_match.group(1)) if lat_match else None,
        throughput_fps=float(thr_match.group(1)) if thr_match else None,
        raw=out,
    )


def run_compare(cwd: Path, compare_script: Path, model: Path, ref_dir: Path, topk: int, env: Dict[str, str]) -> tuple[ErrResult, ErrResult]:
    cmd = [
        sys.executable,
        str(compare_script),
        "--model",
        str(model),
        "--ref-dir",
        str(ref_dir),
        "--topk",
        str(topk),
    ]
    out = run_cmd(cmd, cwd, env)

    cpu = CPU_REF_RE.search(out)
    gpu = GPU_REF_RE.search(out)

    cpu_res = ErrResult(
        max_abs=float(cpu.group(1)) if cpu else None,
        max_rel=float(cpu.group(3)) if cpu else None,
        raw=out,
    )
    gpu_res = ErrResult(
        max_abs=float(gpu.group(1)) if gpu else None,
        max_rel=float(gpu.group(3)) if gpu else None,
        raw=out,
    )
    return cpu_res, gpu_res


def fmt(v: Optional[float], digits: int = 4) -> str:
    if v is None:
        return "-"
    return f"{v:.{digits}f}"


def fmt_sci(v: Optional[float]) -> str:
    if v is None:
        return "-"
    return f"{v:.6e}"


def make_table(cpu_bench: BenchResult, gpu_ref_bench: BenchResult, gpu_opt_bench: BenchResult,
               cpu_err: ErrResult, gpu_ref_err: ErrResult, gpu_opt_err: ErrResult) -> str:
    lines = [
        "| Variant | Device | Latency (ms) | Throughput (FPS) | max_abs | max_rel |",
        "|---|---|---:|---:|---:|---:|",
        f"| ref | GPU | {fmt(gpu_ref_bench.latency_ms)} | {fmt(gpu_ref_bench.throughput_fps)} | {fmt_sci(gpu_ref_err.max_abs)} | {fmt_sci(gpu_ref_err.max_rel)} |",
        f"| opt | GPU | {fmt(gpu_opt_bench.latency_ms)} | {fmt(gpu_opt_bench.throughput_fps)} | {fmt_sci(gpu_opt_err.max_abs)} | {fmt_sci(gpu_opt_err.max_rel)} |",
        f"| ref | CPU | {fmt(cpu_bench.latency_ms)} | {fmt(cpu_bench.throughput_fps)} | {fmt_sci(cpu_err.max_abs)} | {fmt_sci(cpu_err.max_rel)} |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BevPoolV2 performance and parity report (ref vs opt)")
    parser.add_argument("--repo-root", default=".", help="OpenVINO repository root")
    parser.add_argument("--model", required=True, help="Path to model (ONNX/IR)")
    parser.add_argument("--ref-dir", required=True, help="Path to reference bin directory")
    parser.add_argument("--shape", required=True, help="benchmark_app -shape argument")
    parser.add_argument("--benchmark-bin", default="./bin/intel64/Release/benchmark_app", help="benchmark_app binary")
    parser.add_argument("--compare-script", default="./compare_bevpool_ref.py", help="compare script path")
    parser.add_argument("--ov-python-dir", default="./bin/intel64/Release/python", help="Local OpenVINO Python package dir prepended to PYTHONPATH for parity")
    parser.add_argument("--ov-lib-dir", default="./bin/intel64/Release", help="Local OpenVINO runtime lib dir prepended to LD_LIBRARY_PATH for parity")
    parser.add_argument("--niter", type=int, default=100)
    parser.add_argument("--nireq", type=int, default=1)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--skip-parity", action="store_true", help="Skip compare_bevpool_ref.py parity step")
    parser.add_argument("--output", default="./bevpool_performance_parity_report.md", help="Output markdown report")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model = (repo_root / args.model).resolve() if not Path(args.model).is_absolute() else Path(args.model)
    ref_dir = (repo_root / args.ref_dir).resolve() if not Path(args.ref_dir).is_absolute() else Path(args.ref_dir)
    benchmark_bin = (repo_root / args.benchmark_bin).resolve() if not Path(args.benchmark_bin).is_absolute() else Path(args.benchmark_bin)
    compare_script = (repo_root / args.compare_script).resolve() if not Path(args.compare_script).is_absolute() else Path(args.compare_script)
    ov_python_dir = (repo_root / args.ov_python_dir).resolve() if not Path(args.ov_python_dir).is_absolute() else Path(args.ov_python_dir)
    ov_lib_dir = (repo_root / args.ov_lib_dir).resolve() if not Path(args.ov_lib_dir).is_absolute() else Path(args.ov_lib_dir)
    output = (repo_root / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    base_env = os.environ.copy()

    cpu_bench = run_benchmark(repo_root, benchmark_bin, model, "CPU", args.shape, args.niter, args.nireq, base_env)

    env_ref = base_env.copy()
    env_ref["OV_GPU_BEVPOOL_V2_FORCE_REF"] = "1"
    gpu_ref_bench = run_benchmark(repo_root, benchmark_bin, model, "GPU", args.shape, args.niter, args.nireq, env_ref)

    env_opt = base_env.copy()
    env_opt["OV_GPU_BEVPOOL_V2_FORCE_OPT8"] = "1"
    gpu_opt_bench = run_benchmark(repo_root, benchmark_bin, model, "GPU", args.shape, args.niter, args.nireq, env_opt)

    cpu_err_ref = ErrResult(max_abs=None, max_rel=None, raw="")
    gpu_err_ref = ErrResult(max_abs=None, max_rel=None, raw="")
    gpu_err_opt = ErrResult(max_abs=None, max_rel=None, raw="")
    parity_note = "Parity step executed"

    if not args.skip_parity:
        parity_env_ref = env_ref.copy()
        parity_env_opt = env_opt.copy()
        if ov_python_dir.exists():
            prepend_path(parity_env_ref, "PYTHONPATH", str(ov_python_dir))
            prepend_path(parity_env_opt, "PYTHONPATH", str(ov_python_dir))
        if ov_lib_dir.exists():
            prepend_path(parity_env_ref, "LD_LIBRARY_PATH", str(ov_lib_dir))
            prepend_path(parity_env_opt, "LD_LIBRARY_PATH", str(ov_lib_dir))
        try:
            cpu_err_ref, gpu_err_ref = run_compare(repo_root, compare_script, model, ref_dir, args.topk, parity_env_ref)
            _, gpu_err_opt = run_compare(repo_root, compare_script, model, ref_dir, args.topk, parity_env_opt)
        except Exception as exc:
            parity_note = f"Parity step skipped due to error: {exc}"
    else:
        parity_note = "Parity step skipped by --skip-parity"

    table = make_table(cpu_bench, gpu_ref_bench, gpu_opt_bench, cpu_err_ref, gpu_err_ref, gpu_err_opt)

    report = [
        "# BevPoolV2 Performance and Parity Report",
        "",
        f"- Model: {model}",
        f"- Ref dir: {ref_dir}",
        f"- Shape: {args.shape}",
        f"- Iterations: {args.niter}, Requests: {args.nireq}",
        f"- Parity: {parity_note}",
        "",
        table,
        "",
    ]

    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Report written to: {output}")
    print()
    print(table)


if __name__ == "__main__":
    main()
