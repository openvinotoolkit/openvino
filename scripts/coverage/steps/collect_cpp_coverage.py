# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

from coverage_workflow import CoverageContext, run_cmd, warn


def _has_gcda(root: Path) -> bool:
    if not root.exists():
        return False
    return any(root.rglob("*.gcda"))


def _read_header(path: Path) -> bytes | None:
    try:
        with path.open("rb") as f:
            header = f.read(12)
        if len(header) < 12:
            return None
        return header
    except OSError:
        return None


def _prefilter_incompatible_gcda(root: Path, *, label: str) -> None:
    if not root.exists():
        return

    removed_missing_gcno = 0
    removed_header_mismatch = 0
    removed_unreadable = 0
    scanned = 0

    for gcda in root.rglob("*.gcda"):
        scanned += 1
        gcno = gcda.with_suffix(".gcno")
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


def run(ctx: CoverageContext) -> None:
    src_dir = ctx.workspace
    report_dir = ctx.workspace / "coverage-report"
    main_info = ctx.workspace / "coverage-cpp-main.info"
    js_info = ctx.workspace / "coverage-cpp-js.info"
    merged_info = ctx.workspace / "coverage.info"

    _prefilter_incompatible_gcda(ctx.paths.build_dir, label="main build")
    if ctx.paths.build_js_dir.exists():
        _prefilter_incompatible_gcda(ctx.paths.build_js_dir, label="js build")

    has_main_gcda = _has_gcda(ctx.paths.build_dir)
    has_js_gcda = _has_gcda(ctx.paths.build_js_dir)

    if has_main_gcda:
        _run_lcov_capture(
            directory=ctx.paths.build_dir,
            build_directory=ctx.paths.build_dir,
            base_directory=src_dir,
            output_file=main_info,
            label="C/C++ main capture",
        )
    else:
        warn(f"No .gcda files found in {ctx.paths.build_dir}, skipping main native C++ capture")

    if has_js_gcda:
        _run_lcov_capture(
            directory=ctx.paths.build_js_dir,
            build_directory=ctx.paths.build_js_dir,
            base_directory=src_dir,
            output_file=js_info,
            label="C/C++ JS-side capture",
        )

    if main_info.exists() and js_info.exists():
        run_cmd(["lcov", "-a", str(main_info), "-a", str(js_info), "-o", str(merged_info)])
    elif main_info.exists():
        merged_info.write_bytes(main_info.read_bytes())
    elif js_info.exists():
        merged_info.write_bytes(js_info.read_bytes())
    else:
        warn("No native C/C++ coverage tracefiles were produced; creating empty coverage.info")
        merged_info.write_text("", encoding="utf-8")

    if not merged_info.exists() or merged_info.stat().st_size == 0:
        warn("coverage.info is empty; skipping lcov --remove and genhtml.")
        if main_info.exists():
            main_info.unlink()
        if js_info.exists():
            js_info.unlink()
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

    if main_info.exists():
        main_info.unlink()
    if js_info.exists():
        js_info.unlink()

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
