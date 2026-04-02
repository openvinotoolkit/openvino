# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from coverage_workflow import CoverageContext, run_cmd, warn


def _slugify(value: str) -> str:
    """Convert a label into a filesystem-friendly fragment."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-") or "capture"


def _as_text(value: str | bytes | None) -> str:
    """Normalize subprocess output to text."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


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
    does not break capture for the whole run.
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


def _classify_unwanted_gcda(rel_path: Path) -> str | None:
    """Return the exclusion reason for a gcda file we never want to capture."""
    rel = rel_path.as_posix()
    padded = f"/{rel}/"

    if rel.endswith(".pb.cc.gcda") or rel.endswith(".pb.h.gcda"):
        return "generated-protobuf"
    if rel.startswith("thirdparty/") or "/thirdparty/" in padded:
        return "thirdparty"
    if "/tests/" in padded or rel.startswith("tests/"):
        return "tests"
    if rel.startswith("docs/") or "/docs/" in padded:
        return "docs"
    if rel.startswith("samples/") or "/samples/" in padded:
        return "samples"
    if rel.startswith("tools/") or "/tools/" in padded:
        return "tools"
    return None


def _prune_unwanted_gcda(root: Path, *, label: str) -> None:
    """Delete gcda files that are known to be excluded from the final report.

    This reduces lcov capture time by removing whole classes of files that we
    already strip later via ``lcov --remove`` and never want to upload.
    """
    if not root.exists():
        return

    removed = 0
    reasons: dict[str, int] = {}
    for gcda in root.rglob("*.gcda"):
        reason = _classify_unwanted_gcda(gcda.relative_to(root))
        if reason is None:
            continue
        try:
            gcda.unlink()
            removed += 1
            reasons[reason] = reasons.get(reason, 0) + 1
        except OSError:
            continue

    if removed:
        details = ", ".join(f"{name}={count}" for name, count in sorted(reasons.items()))
        warn(f"{label}: pre-pruned {removed} unwanted gcda files before lcov capture ({details})")


def _tool_version_report() -> str:
    """Capture native coverage tool versions for debugging."""
    commands = (
        ("gcc", ["gcc", "--version"]),
        ("g++", ["g++", "--version"]),
        ("gcov", ["gcov", "--version"]),
        ("lcov", ["lcov", "--version"]),
        ("genhtml", ["genhtml", "--version"]),
    )

    lines: list[str] = []
    for name, cmd in commands:
        resolved = shutil.which(name)
        lines.append(f"## {name}")
        lines.append(f"path: {resolved or 'not found'}")
        if resolved is None:
            lines.append("")
            continue
        completed = subprocess.run(cmd, text=True, capture_output=True)
        lines.append(f"exit_code: {completed.returncode}")
        output = (completed.stdout or completed.stderr or "").strip()
        if output:
            lines.extend(output.splitlines()[:3])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _tree_inventory_report(
    *,
    root: Path,
    label: str,
    gcno_root: Path,
    alt_gcno_root: Path | None = None,
    limit: int = 20,
) -> str:
    """Summarize gcda/gcno layout and sample expected note matches."""
    lines = [f"# {label}", f"root={root}", f"gcno_root={gcno_root}"]
    if alt_gcno_root is not None:
        lines.append(f"alt_gcno_root={alt_gcno_root}")

    if not root.exists():
        lines.append("exists=false")
        return "\n".join(lines) + "\n"

    local_gcno_files = sorted(root.rglob("*.gcno"))
    gcda_files = sorted(root.rglob("*.gcda"))
    lines.extend(
        [
            "exists=true",
            f"gcno_count={len(local_gcno_files)}",
            f"gcda_count={len(gcda_files)}",
        ]
    )

    if gcda_files:
        local_matches = 0
        alt_matches = 0
        missing_matches = 0
        lines.append("sample_gcda_to_gcno:")
        for gcda in gcda_files:
            rel = gcda.relative_to(root)
            local_gcno = gcno_root / rel.with_suffix(".gcno")
            alt_gcno = alt_gcno_root / rel.with_suffix(".gcno") if alt_gcno_root is not None else None
            local_exists = local_gcno.exists()
            alt_exists = alt_gcno.exists() if alt_gcno is not None else False
            if local_exists:
                local_matches += 1
            elif alt_exists:
                alt_matches += 1
            else:
                missing_matches += 1

        lines.extend(
            [
                f"matching_gcno_in_primary_root={local_matches}",
                f"matching_gcno_only_in_alt_root={alt_matches}",
                f"matching_gcno_missing_in_both={missing_matches}",
            ]
        )

        for gcda in gcda_files[:limit]:
            rel = gcda.relative_to(root)
            local_gcno = gcno_root / rel.with_suffix(".gcno")
            alt_gcno = alt_gcno_root / rel.with_suffix(".gcno") if alt_gcno_root is not None else None
            lines.append(
                "  "
                + f"gcda={gcda} | primary_gcno_exists={local_gcno.exists()} | primary_gcno={local_gcno}"
                + (f" | alt_gcno_exists={alt_gcno.exists()} | alt_gcno={alt_gcno}" if alt_gcno is not None else "")
            )

    if local_gcno_files:
        lines.append("sample_gcno:")
        lines.extend(f"  {path}" for path in local_gcno_files[:limit])

    return "\n".join(lines) + "\n"


def _run_lcov_capture(
    *,
    directory: Path,
    base_directory: Path,
    output_file: Path,
    label: str,
    debug_dir: Path,
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
        "--base-directory",
        str(base_directory),
        "--output-file",
        str(output_file),
        "--no-external",
        "--rc",
        "geninfo_unexecuted_blocks=1",
        "--ignore-errors",
        "gcov,source,graph",
        "--compat",
        "split_crc=auto",
    ]

    timeout_seconds = int(os.environ.get("LCOV_CAPTURE_TIMEOUT_SECONDS", "3600"))
    max_attempts = 32
    for attempt in range(1, max_attempts + 1):
        display = " ".join(shlex.quote(part) for part in cmd)
        log_file = debug_dir / f"{_slugify(label)}.attempt{attempt}.log"
        print(
            f"[coverage] $ {display}  # {label} "
            f"(attempt {attempt}/{max_attempts}, timeout={timeout_seconds}s)"
        )
        try:
            completed = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            stdout_text = _as_text(exc.stdout)
            stderr_text = _as_text(exc.stderr)
            log_file.write_text(
                f"$ {display}\nexit_code=timeout\n\n[stdout]\n{stdout_text}\n\n[stderr]\n{stderr_text}\n",
                encoding="utf-8",
            )
            combined_tail = (stdout_text.splitlines() + stderr_text.splitlines())[-40:]
            if combined_tail:
                warn(
                    f"{label}: lcov capture timed out after {timeout_seconds}s; full log: {log_file}\n"
                    + "\n".join(combined_tail)
                )
            raise RuntimeError(f"{label}: lcov capture timed out after {timeout_seconds}s") from exc

        stdout_text = _as_text(completed.stdout)
        stderr_text = _as_text(completed.stderr)
        log_file.write_text(
            f"$ {display}\nexit_code={completed.returncode}\n\n[stdout]\n{stdout_text}\n\n[stderr]\n{stderr_text}\n",
            encoding="utf-8",
        )

        if completed.returncode == 0 and output_file.exists() and output_file.stat().st_size > 0:
            return

        log_text = f"{stdout_text}\n{stderr_text}"
        bad_files = _extract_problematic_gcda(log_text)
        removed = _remove_gcda_files(bad_files)

        if removed > 0:
            preview = ", ".join(str(p) for p in bad_files[:3])
            more = "" if len(bad_files) <= 3 else f", ... (+{len(bad_files) - 3} more)"
            warn(f"{label}: removed {removed} problematic .gcda file(s): {preview}{more}; lcov log: {log_file}")
            continue

        finished_info = "Finished .info-file creation" in log_text
        excluded_match = re.search(r"Excluded data for (\d+) files due to include/exclude options", log_text)
        excluded_suffix = f", excluded={excluded_match.group(1)}" if excluded_match else ""
        if output_file.exists():
            size_bytes = output_file.stat().st_size
        else:
            size_bytes = -1

        if output_file.exists() and (size_bytes > 0 or finished_info):
            warn(
                f"{label}: lcov returned {completed.returncode}, but {output_file.name} was produced "
                f"(size={size_bytes} bytes{excluded_suffix}); continuing. lcov log: {log_file}"
            )
            return

        combined_tail = (stdout_text.splitlines() + stderr_text.splitlines())[-40:]
        if combined_tail:
            warn(
                f"{label}: lcov capture failed (attempt {attempt}); full log: {log_file}\n"
                + "\n".join(combined_tail)
            )
        raise RuntimeError(f"{label}: lcov capture failed without recoverable gcda cleanup; see {log_file}")

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

    tool_report = _tool_version_report()
    (trace_dir / "tool-versions.txt").write_text(tool_report, encoding="utf-8")
    print("[coverage] Native coverage tool versions:")
    print(tool_report.rstrip())

    (trace_dir / "main-build-before-prefilter.txt").write_text(
        _tree_inventory_report(
            root=ctx.paths.build_dir,
            label="main build before prefilter",
            gcno_root=ctx.paths.build_dir,
            alt_gcno_root=ctx.paths.build_js_dir if ctx.paths.build_js_dir.exists() else None,
        ),
        encoding="utf-8",
    )
    (trace_dir / "js-build-before-prefilter.txt").write_text(
        _tree_inventory_report(
            root=ctx.paths.build_js_dir,
            label="js build before prefilter",
            gcno_root=ctx.paths.build_js_dir,
            alt_gcno_root=ctx.paths.build_dir if ctx.paths.build_dir.exists() else None,
        ),
        encoding="utf-8",
    )

    _prune_unwanted_gcda(ctx.paths.build_dir, label="main build")
    if ctx.paths.build_js_dir.exists():
        _prune_unwanted_gcda(ctx.paths.build_js_dir, label="js build")

    _prefilter_incompatible_gcda(ctx.paths.build_dir, label="main build")
    if ctx.paths.build_js_dir.exists():
        _prefilter_incompatible_gcda(ctx.paths.build_js_dir, label="js build")

    (trace_dir / "main-build-after-prefilter.txt").write_text(
        _tree_inventory_report(
            root=ctx.paths.build_dir,
            label="main build after prefilter",
            gcno_root=ctx.paths.build_dir,
            alt_gcno_root=ctx.paths.build_js_dir if ctx.paths.build_js_dir.exists() else None,
        ),
        encoding="utf-8",
    )
    (trace_dir / "js-build-after-prefilter.txt").write_text(
        _tree_inventory_report(
            root=ctx.paths.build_js_dir,
            label="js build after prefilter",
            gcno_root=ctx.paths.build_js_dir,
            alt_gcno_root=ctx.paths.build_dir if ctx.paths.build_dir.exists() else None,
        ),
        encoding="utf-8",
    )
    print(f"[coverage] Wrote native coverage inventories under {trace_dir}")
    for inventory_name in (
        "main-build-before-prefilter.txt",
        "main-build-after-prefilter.txt",
        "js-build-before-prefilter.txt",
        "js-build-after-prefilter.txt",
    ):
        inventory_path = trace_dir / inventory_name
        if inventory_path.is_file():
            print(f"[coverage] ===== {inventory_name} =====")
            print("\n".join(inventory_path.read_text(encoding="utf-8").splitlines()[:40]))

    has_main_gcda = _has_gcda(ctx.paths.build_dir)
    has_js_gcda = _has_gcda(ctx.paths.build_js_dir)

    if has_main_gcda:
        main_info = trace_dir / "main-build.info"
        _run_lcov_capture(
            directory=ctx.paths.build_dir,
            base_directory=src_dir,
            output_file=main_info,
            label="C/C++ main capture",
            debug_dir=trace_dir,
        )
    else:
        if not has_staged_main_gcda:
            warn(f"No .gcda files found in {ctx.paths.build_dir}, skipping main native C++ capture")

        main_info = None

    if has_js_gcda:
        js_info = trace_dir / "js-build.info"
        _run_lcov_capture(
            directory=ctx.paths.build_js_dir,
            base_directory=src_dir,
            output_file=js_info,
            label="C/C++ JS-side capture",
            debug_dir=trace_dir,
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
                base_directory=src_dir,
                output_file=tracefile,
                label=f"C/C++ staged main capture ({run_dir.name})",
                debug_dir=trace_dir,
            )
            tracefiles.append(tracefile)

        staged_js_dir = run_dir / "build_js"
        if _has_gcda(staged_js_dir):
            _prefilter_incompatible_gcda(staged_js_dir, label=f"staged js build ({run_dir.name})", gcno_root=ctx.paths.build_js_dir)
            tracefile = trace_dir / f"{run_dir.name}-js.info"
            _run_lcov_capture(
                directory=staged_js_dir,
                base_directory=src_dir,
                output_file=tracefile,
                label=f"C/C++ staged JS capture ({run_dir.name})",
                debug_dir=trace_dir,
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
            "unused",
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
