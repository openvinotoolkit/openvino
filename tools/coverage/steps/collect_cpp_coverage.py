# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shlex
import shutil
import subprocess
from pathlib import Path

from coverage_workflow import CoverageContext, run_cmd, warn


def _has_gcda(root: Path) -> bool:
    """Return whether a directory tree contains any gcda files."""
    if not root.exists():
        return False
    return any(root.rglob("*.gcda"))


def _read_header(path: Path) -> bytes | None:
    """Read the 12-byte gcov header used for compatibility checks.

    Example layout: ``<4-byte magic><4-byte version><4-byte stamp/checksum>``.
    This helper only returns the raw 12 bytes so callers can compare fields such
    as ``header[4:12]`` between matching ``.gcda`` and ``.gcno`` files. This is
    needed because stale runtime data from an older build can make ``lcov`` fail
    with mismatch errors even when the current build is otherwise valid.
    """
    try:
        with path.open("rb") as f:
            header = f.read(12)
        if len(header) < 12:
            return None
        return header
    except OSError:
        return None


def _prefilter_incompatible_gcda(root: Path, *, label: str, gcno_root: Path | None = None) -> None:
    """Remove gcda files that cannot be matched to compatible gcno files.

    This is a proactive cleanup step before running ``lcov``. It filters out
    stale, unreadable, or mismatched coverage data so one bad ``.gcda`` file
    does not break capture for the whole shard.
    """
    if not root.exists():
        return

    removed_missing_gcno = 0
    removed_header_mismatch = 0
    removed_unreadable = 0
    scanned = 0

    for gcda in root.rglob("*.gcda"):
        scanned += 1
        if gcno_root is None:
            gcno = gcda.with_suffix(".gcno")
        else:
            gcno = gcno_root / gcda.relative_to(root).with_suffix(".gcno")
        if not gcno.exists():
            try:
                gcda.unlink()
                removed_missing_gcno += 1
            except OSError:
                pass
            continue

        gcda_header = _read_header(gcda)
        gcno_header = _read_header(gcno)
        if gcda_header is None or gcno_header is None:
            try:
                gcda.unlink()
                removed_unreadable += 1
            except OSError:
                pass
            continue

        # bytes[4:12] are version+stamp/checksum fields that must match between gcda/gcno.
        if gcda_header[4:12] != gcno_header[4:12]:
            try:
                gcda.unlink()
                removed_header_mismatch += 1
            except OSError:
                pass

    if removed_missing_gcno or removed_header_mismatch or removed_unreadable:
        warn(
            f"{label}: pre-filtered incompatible gcda files "
            f"(scanned={scanned}, missing_gcno={removed_missing_gcno}, "
            f"header_mismatch={removed_header_mismatch}, unreadable={removed_unreadable})"
        )


def _extract_problematic_gcda(log_text: str) -> list[Path]:
    """Extract gcda paths mentioned in lcov/gcov error output.

    This is used after a failed ``lcov`` attempt to identify the specific
    runtime coverage files that triggered mismatch or missing-notes errors, so
    they can be removed and capture can be retried.
    """
    patterns = (
        r"(?m)^([^\n]+\.gcda):stamp mismatch with notes file$",
        r"GCOV failed for ([^\s!]+\.gcda)!",
        r"skipping \.gcda file ([^\s]+\.gcda) because corresponding \.gcno file",
    )

    found: set[Path] = set()
    for pattern in patterns:
        for raw in re.findall(pattern, log_text):
            path = Path(raw.strip())
            if path.exists():
                found.add(path)
    return sorted(found)


def _remove_gcda_files(paths: list[Path]) -> int:
    """Delete problematic gcda files and return how many were removed.

    Removing only the known-bad files lets coverage collection continue with the
    remaining valid data instead of failing the entire report.
    """
    removed = 0
    for path in paths:
        try:
            path.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def _run_lcov_capture(
    *,
    directory: Path,
    build_directory: Path,
    base_directory: Path,
    output_file: Path,
    label: str,
) -> None:
    """Capture one lcov tracefile, retrying after gcda cleanup when needed.

    ``lcov`` can fail on mismatched or corrupted ``.gcda`` inputs. This helper
    wraps capture with targeted cleanup and retry logic so transient artifact or
    stale-file issues do not abort native coverage generation unnecessarily.
    """
    cmd = [
        "lcov",
        "--capture",
        "--directory",
        str(directory),
        "--build-directory",
        str(build_directory),
        "--base-directory",
        str(base_directory),
        "--output-file",
        str(output_file),
        "--no-external",
        "--rc",
        "geninfo_unexecuted_blocks=1",
        "--ignore-errors",
        "gcov,source,graph,mismatch",
        "--compat",
        "split_crc=auto",
    ]

    max_attempts = 32
    for attempt in range(1, max_attempts + 1):
        display = " ".join(shlex.quote(part) for part in cmd)
        print(f"[coverage] $ {display}  # {label} (attempt {attempt}/{max_attempts})")
        completed = subprocess.run(cmd, text=True, capture_output=True)

        if completed.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            return

        log_text = f"{completed.stdout or ''}\n{completed.stderr or ''}"
        bad_files = _extract_problematic_gcda(log_text)
        removed = _remove_gcda_files(bad_files)

        if removed > 0:
            preview = ", ".join(str(p) for p in bad_files[:3])
            more = "" if len(bad_files) <= 3 else f", ... (+{len(bad_files) - 3} more)"
            warn(f"{label}: removed {removed} problematic .gcda file(s): {preview}{more}")
            continue

        if output_file.exists() and output_file.stat().st_size > 0:
            warn(f"{label}: lcov returned {completed.returncode}, but {output_file.name} was produced; continuing.")
            return

        stderr_tail = (completed.stderr or "").strip().splitlines()[-8:]
        if stderr_tail:
            warn(f"{label}: lcov capture failed (attempt {attempt}) with tail:\n" + "\n".join(stderr_tail))

    raise RuntimeError(f"{label}: failed to capture coverage after {max_attempts} attempts")


def _collect_staged_gcov_runs(ctx: CoverageContext) -> list[Path]:
    """List staged gcov run directories created by parallel C++ tests.

    Parallel C++ test execution writes gcov data into isolated staging
    directories to avoid file collisions, and those directories must later be
    discovered and merged into the final coverage report.
    """
    runs_root = ctx.workspace / ".tmp" / "cpp-gcov" / "runs"
    if not runs_root.exists():
        return []
    return sorted(path for path in runs_root.iterdir() if path.is_dir())


def _merge_tracefiles(tracefiles: list[Path], output: Path) -> None:
    """Merge captured lcov tracefiles into the final coverage report.

    Coverage may be captured from the main build, the JS build, and staged
    parallel-test runs, so those partial reports need to be combined into a
    single ``coverage.info`` file for downstream upload and HTML generation.
    """
    if not tracefiles:
        warn("No native C/C++ coverage tracefiles were produced; creating empty coverage.info")
        output.write_text("", encoding="utf-8")
        return

    if len(tracefiles) == 1:
        output.write_bytes(tracefiles[0].read_bytes())
        return

    cmd = ["lcov"]
    for tracefile in tracefiles:
        cmd.extend(["-a", str(tracefile)])
    cmd.extend(["-o", str(output)])
    run_cmd(cmd)


def run(ctx: CoverageContext) -> None:
    """Collect native C/C++ coverage data and generate HTML output.

    The flow intentionally cleans up incompatible gcov artifacts, captures all
    available native tracefiles, merges them, and then produces the final report
    used by Codecov and local HTML inspection.
    """
    src_dir = ctx.workspace
    report_dir = ctx.workspace / "coverage-report"
    merged_info = ctx.workspace / "coverage.info"
    trace_dir = ctx.workspace / ".tmp" / "cpp-coverage-parts"
    shutil.rmtree(trace_dir, ignore_errors=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    tracefiles: list[Path] = []
    staged_runs = _collect_staged_gcov_runs(ctx)
    has_staged_main_gcda = any(_has_gcda(run_dir / "build") for run_dir in staged_runs)
    has_staged_js_gcda = any(_has_gcda(run_dir / "build_js") for run_dir in staged_runs)

    _prefilter_incompatible_gcda(ctx.paths.build_dir, label="main build")
    if ctx.paths.build_js_dir.exists():
        _prefilter_incompatible_gcda(ctx.paths.build_js_dir, label="js build")

    has_main_gcda = _has_gcda(ctx.paths.build_dir)
    has_js_gcda = _has_gcda(ctx.paths.build_js_dir)

    if has_main_gcda:
        main_info = trace_dir / "main-build.info"
        _run_lcov_capture(
            directory=ctx.paths.build_dir,
            build_directory=ctx.paths.build_dir,
            base_directory=src_dir,
            output_file=main_info,
            label="C/C++ main capture",
        )
    else:
        if not has_staged_main_gcda:
            warn(f"No .gcda files found in {ctx.paths.build_dir}, skipping main native C++ capture")

        main_info = None

    if has_js_gcda:
        js_info = trace_dir / "js-build.info"
        _run_lcov_capture(
            directory=ctx.paths.build_js_dir,
            build_directory=ctx.paths.build_js_dir,
            base_directory=src_dir,
            output_file=js_info,
            label="C/C++ JS-side capture",
        )
        tracefiles.append(js_info)
    else:
        if not has_staged_js_gcda and ctx.paths.build_js_dir.exists():
            warn(f"No .gcda files found in {ctx.paths.build_js_dir}, skipping JS-side native C++ capture")
        js_info = None

    if has_main_gcda and main_info is not None:
        tracefiles.append(main_info)

    if staged_runs:
        print(f"[coverage] Found {len(staged_runs)} staged C++ gcov run(s) under {ctx.workspace / '.tmp' / 'cpp-gcov' / 'runs'}")

    for run_dir in staged_runs:
        staged_main_dir = run_dir / "build"
        if _has_gcda(staged_main_dir):
            _prefilter_incompatible_gcda(staged_main_dir, label=f"staged main build ({run_dir.name})", gcno_root=ctx.paths.build_dir)
            tracefile = trace_dir / f"{run_dir.name}-main.info"
            _run_lcov_capture(
                directory=staged_main_dir,
                build_directory=ctx.paths.build_dir,
                base_directory=src_dir,
                output_file=tracefile,
                label=f"C/C++ staged main capture ({run_dir.name})",
            )
            tracefiles.append(tracefile)

        staged_js_dir = run_dir / "build_js"
        if _has_gcda(staged_js_dir):
            _prefilter_incompatible_gcda(staged_js_dir, label=f"staged js build ({run_dir.name})", gcno_root=ctx.paths.build_js_dir)
            tracefile = trace_dir / f"{run_dir.name}-js.info"
            _run_lcov_capture(
                directory=staged_js_dir,
                build_directory=ctx.paths.build_js_dir,
                base_directory=src_dir,
                output_file=tracefile,
                label=f"C/C++ staged JS capture ({run_dir.name})",
            )
            tracefiles.append(tracefile)

    _merge_tracefiles(tracefiles, merged_info)

    if not merged_info.exists() or merged_info.stat().st_size == 0:
        warn("coverage.info is empty; skipping lcov --remove and genhtml.")
        shutil.rmtree(trace_dir, ignore_errors=True)
        return

    run_cmd(
        [
            "lcov",
            "--remove",
            str(merged_info),
            "--ignore-errors",
            "unused,mismatch",
            f"{src_dir}/build/*",
            f"{src_dir}/build_js/*",
            f"{src_dir}/*.pb.cc",
            f"{src_dir}/*.pb.h",
            f"{src_dir}/*/tests/*",
            f"{src_dir}/tests/*",
            f"{src_dir}/docs/*",
            f"{src_dir}/samples/*",
            f"{src_dir}/tools/*",
            f"{src_dir}/src/bindings/js/node/tests/*",
            f"{src_dir}/src/bindings/python/tests/*",
            f"{src_dir}/thirdparty/*",
            "-o",
            str(merged_info),
        ]
    )

    shutil.rmtree(trace_dir, ignore_errors=True)

    run_cmd(["grep", "-m", "5", f"^SF:{src_dir}/build/", str(merged_info)], check=False)
    run_cmd(["grep", "-m", "5", f"^SF:{src_dir}/build_js/", str(merged_info)], check=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    genhtml_rc = run_cmd(
        [
            "genhtml",
            str(merged_info),
            "--output-directory",
            str(report_dir),
            "--prefix",
            str(src_dir),
            "--synthesize-missing",
        ],
        check=False,
    )
    if genhtml_rc != 0:
        warn("genhtml failed; coverage.info is still available for Codecov upload.")
