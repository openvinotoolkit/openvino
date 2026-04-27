#!/usr/bin/env python3
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class ShapeSpec:
    args: List[str]
    label: str


@dataclass
class ResultRow:
    kind: str
    shape: str
    src: str
    kernel: str
    ms: Optional[float]
    gops: Optional[float]
    note: str


@dataclass
class SummaryRow:
    shape: str
    src: str
    our_kernel: str
    our_ms: Optional[float]
    our_pack_ms: Optional[float]
    our_compute_ms: Optional[float]
    acl_ms: Optional[float]
    kleidiai_total_ms: Optional[float]
    kleidiai_compute_ms: Optional[float]
    speedup_acl: Optional[float]
    speedup_kleidiai_total: Optional[float]
    speedup_kleidiai_compute: Optional[float]
    gap_note: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def default_bin_path() -> Path:
    return repo_root() / "bin" / "aarch64" / "Release" / "ov_cpu_int8_microbench"


def default_gemm_shapes() -> List[ShapeSpec]:
    return [
        ShapeSpec(args=["--m=64", "--n=64", "--k=128"], label="M64xN64xK128"),
        ShapeSpec(args=["--m=128", "--n=64", "--k=128"], label="M128xN64xK128"),
        ShapeSpec(args=["--m=128", "--n=128", "--k=128"], label="M128xN128xK128"),
        ShapeSpec(args=["--m=64", "--n=96", "--k=128"], label="M64xN96xK128"),
        ShapeSpec(args=["--m=96", "--n=96", "--k=128"], label="M96xN96xK128"),
        ShapeSpec(args=["--m=128", "--n=96", "--k=64"], label="M128xN96xK64"),
    ]


def default_conv_shapes() -> List[ShapeSpec]:
    return [
        ShapeSpec(
            args=["--N=1", "--H=28", "--W=28", "--IC=64", "--OC=64", "--KH=1", "--KW=1", "--stride=1"],
            label="N1H28W28IC64OC64K1x1S1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=28", "--W=28", "--IC=64", "--OC=128", "--KH=1", "--KW=1", "--stride=1"],
            label="N1H28W28IC64OC128K1x1S1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=28", "--W=28", "--IC=64", "--OC=128", "--KH=3", "--KW=3", "--stride=1"],
            label="N1H28W28IC64OC128K3x3S1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=64",
                "--OC=128",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=1",
            ],
            label="N1H28W28IC64OC128K3x3S1P1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=28", "--W=28", "--IC=64", "--OC=64", "--KH=5", "--KW=5", "--stride=1"],
            label="N1H28W28IC64OC64K5x5S1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=64",
                "--OC=64",
                "--KH=5",
                "--KW=5",
                "--stride=1",
                "--pad=2",
            ],
            label="N1H28W28IC64OC64K5x5S1P2",
        ),
        ShapeSpec(
            args=["--N=1", "--H=56", "--W=56", "--IC=32", "--OC=64", "--KH=3", "--KW=3", "--stride=1"],
            label="N1H56W56IC32OC64K3x3S1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=56",
                "--W=56",
                "--IC=32",
                "--OC=64",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=1",
            ],
            label="N1H56W56IC32OC64K3x3S1P1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=14", "--W=14", "--IC=128", "--OC=128", "--KH=1", "--KW=1", "--stride=1"],
            label="N1H14W14IC128OC128K1x1S1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=14", "--W=14", "--IC=64", "--OC=128", "--KH=3", "--KW=3", "--stride=1"],
            label="N1H14W14IC64OC128K3x3S1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=14",
                "--W=14",
                "--IC=64",
                "--OC=128",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=1",
            ],
            label="N1H14W14IC64OC128K3x3S1P1",
        ),
        ShapeSpec(
            args=["--N=1", "--H=28", "--W=28", "--IC=128", "--OC=128", "--KH=1", "--KW=1", "--stride=1"],
            label="N1H28W28IC128OC128K1x1S1",
        ),
        # Stride-2 downsampling cases.
        ShapeSpec(
            args=[
                "--N=1",
                "--H=56",
                "--W=56",
                "--IC=32",
                "--OC=64",
                "--KH=3",
                "--KW=3",
                "--stride=2",
                "--pad=1",
            ],
            label="N1H56W56IC32OC64K3x3S2P1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=64",
                "--OC=128",
                "--KH=3",
                "--KW=3",
                "--stride=2",
                "--pad=1",
            ],
            label="N1H28W28IC64OC128K3x3S2P1",
        ),
        # Dilated conv (common in segmentation backbones).
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=64",
                "--OC=64",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=2",
                "--dilation=2",
            ],
            label="N1H28W28IC64OC64K3x3S1P2D2",
        ),
        # Grouped conv / depthwise markers (expected unsupported for some backends today).
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=64",
                "--OC=64",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=1",
                "--groups=2",
            ],
            label="N1H28W28IC64OC64G2K3x3S1P1",
        ),
        ShapeSpec(
            args=[
                "--N=1",
                "--H=28",
                "--W=28",
                "--IC=32",
                "--OC=32",
                "--KH=3",
                "--KW=3",
                "--stride=1",
                "--pad=1",
                "--groups=32",
            ],
            label="N1H28W28IC32OC32G32K3x3S1P1_DW",
        ),
    ]


def run_microbench(bin_path: Path, mode: str, shape_args: Iterable[str], signed_src: bool) -> str:
    cmd = [str(bin_path), f"--{mode}"] + list(shape_args)
    if signed_src:
        cmd.append("--signed-src")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    if result.returncode != 0:
        print(f"warning: microbench exited with code {result.returncode} for {' '.join(cmd)}")
    return result.stdout


GEMM_HDR_RE = re.compile(r"^GEMM M=(\d+) N=(\d+) K=(\d+)")
CONV_HDR_RE = re.compile(
    r"^CONV N=(\d+) H=(\d+) W=(\d+) IC=(\d+) OC=(\d+) KH=(\d+) KW=(\d+) stride=(\d+)"
)
LINE_RE = re.compile(r"^\s*(.+?)\s*:\s*([0-9.]+) ms,\s*([0-9.]+) GOPS$")
PACK_ONLY_RE = re.compile(r"^\s*(.+?)\s*:\s*([0-9.]+) ms$")
LABEL_RE = re.compile(r"^[A-Za-z0-9_]+$")
RAW_LABEL_RE = re.compile(r"\[\s*([A-Za-z0-9_]+)\s*\]$")
ALLOWED_PREFIXES = ("our_", "acl_", "kleidiai_")


def src_pair_tag(src: str) -> str:
    return "s8s8" if src == "s8" else "u8s8"


def canonical_kernel_name(kernel: str, src: str) -> str:
    if kernel.startswith("acl_"):
        suffix = kernel[len("acl_") :]
        return f"acl::{suffix}"
    if kernel.startswith("kleidiai_"):
        suffix = kernel[len("kleidiai_") :]
        return f"kleidiai::{suffix}"
    if not kernel.startswith("our_"):
        return kernel

    pair = src_pair_tag(src)

    if kernel == "our_brgemm4x4":
        return "brgemm_wrapper_runtime_dispatch"
    if "_brgemm" in kernel:
        tail = kernel[len("our_") :]
        return f"brgemm_wrapper::{tail}"

    family = "aarch64_neon_mla"
    if "mmla" in kernel:
        family = "aarch64_neon_i8mm"
    elif "dot" in kernel:
        family = "aarch64_neon_dotprod"

    body = kernel[len("our_") :]
    body = body.replace("our_", "")
    body = body.replace("_pack_only", "_pack")
    body = body.replace("_exec_u8", "_exec_bias_comp")
    body = body.replace("_total", "_total")
    return f"{family}_{pair}_{body}"


def display_kernel_name(kernel: str, src: str) -> str:
    canonical = canonical_kernel_name(kernel, src)
    if canonical == kernel:
        return kernel
    return f"{canonical} [{kernel}]"


def parse_output(kind: str, shape_label: str, src_label: str, text: str) -> List[ResultRow]:
    rows: List[ResultRow] = []
    current_kind = None

    def extract_raw_label(label: str) -> Optional[str]:
        label = label.strip()
        if LABEL_RE.match(label) and label.startswith(ALLOWED_PREFIXES):
            return label
        m = RAW_LABEL_RE.search(label)
        if m and m.group(1).startswith(ALLOWED_PREFIXES):
            return m.group(1)
        return None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if GEMM_HDR_RE.match(line):
            current_kind = "GEMM"
            continue
        if CONV_HDR_RE.match(line):
            current_kind = "CONV"
            continue
        if current_kind != kind:
            continue
        m = LINE_RE.match(line)
        if m:
            label = extract_raw_label(m.group(1))
            if not label:
                continue
            rows.append(
                ResultRow(
                    kind=kind,
                    shape=shape_label,
                    src=src_label,
                    kernel=label,
                    ms=float(m.group(2)),
                    gops=float(m.group(3)),
                    note="",
                )
            )
            continue
        m = PACK_ONLY_RE.match(line)
        if m:
            label = extract_raw_label(m.group(1))
            if not label:
                continue
            rows.append(
                ResultRow(
                    kind=kind,
                    shape=shape_label,
                    src=src_label,
                    kernel=label,
                    ms=float(m.group(2)),
                    gops=None,
                    note="",
                )
            )
            continue
        if ":" in line:
            kernel, tail = [part.strip() for part in line.split(":", 1)]
            kernel = extract_raw_label(kernel)
            if not kernel:
                continue
            rows.append(
                ResultRow(
                    kind=kind,
                    shape=shape_label,
                    src=src_label,
                    kernel=kernel,
                    ms=None,
                    gops=None,
                    note=tail,
                )
            )
    return rows


def format_markdown_table(headers: List[str], data: List[List[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in data)
    return "\n".join(out)


def format_table(rows: List[ResultRow]) -> str:
    headers = ["Shape", "Src", "Kernel", "ms", "GOPS", "Note"]
    data: List[List[str]] = []
    for row in rows:
        ms = f"{row.ms:.3f}" if row.ms is not None else "n/a"
        gops = f"{row.gops:.3f}" if row.gops is not None else "n/a"
        data.append([row.shape, row.src, display_kernel_name(row.kernel, row.src), ms, gops, row.note])
    return format_markdown_table(headers, data)


def format_summary_table(rows: List[SummaryRow]) -> str:
    headers = [
        "Shape",
        "Src",
        "Our best",
        "Our ms",
        "Our pack ms",
        "Our compute ms",
        "ACL ms",
        "KleidiAI total ms",
        "KleidiAI compute ms",
        "ACL speedup",
        "KleidiAI total speedup",
        "KleidiAI compute speedup",
        "Gap note",
    ]
    data: List[List[str]] = []
    for row in rows:
        our_ms = f"{row.our_ms:.3f}" if row.our_ms is not None else "n/a"
        our_pack_ms = f"{row.our_pack_ms:.3f}" if row.our_pack_ms is not None else "n/a"
        our_compute_ms = f"{row.our_compute_ms:.3f}" if row.our_compute_ms is not None else "n/a"
        acl_ms = f"{row.acl_ms:.3f}" if row.acl_ms is not None else "n/a"
        kleidiai_total_ms = f"{row.kleidiai_total_ms:.3f}" if row.kleidiai_total_ms is not None else "n/a"
        kleidiai_compute_ms = (
            f"{row.kleidiai_compute_ms:.3f}" if row.kleidiai_compute_ms is not None else "n/a"
        )
        acl_speedup = f"{row.speedup_acl:.2f}x" if row.speedup_acl is not None else "n/a"
        kleidiai_total_speedup = (
            f"{row.speedup_kleidiai_total:.2f}x" if row.speedup_kleidiai_total is not None else "n/a"
        )
        kleidiai_compute_speedup = (
            f"{row.speedup_kleidiai_compute:.2f}x" if row.speedup_kleidiai_compute is not None else "n/a"
        )
        data.append(
            [
                row.shape,
                row.src,
                display_kernel_name(row.our_kernel, row.src) if row.our_kernel else "n/a",
                our_ms,
                our_pack_ms,
                our_compute_ms,
                acl_ms,
                kleidiai_total_ms,
                kleidiai_compute_ms,
                acl_speedup,
                kleidiai_total_speedup,
                kleidiai_compute_speedup,
                row.gap_note,
            ]
        )
    return format_markdown_table(headers, data)


def write_table(path: Path, title: str, rows: List[ResultRow], summary: Optional[List[SummaryRow]]) -> None:
    content = [title]
    if title.startswith("CONV"):
        content.append(
            "Note: KleidiAI conv table now shows both total (materialization + lhs pack + matmul) and "
            "compute-only timings; compare total-to-total and compute-to-compute."
        )
    elif title.startswith("BRGEMM"):
        content.append(
            "Note: KleidiAI GEMM table now shows both total (lhs pack + matmul) and compute-only timings; "
            "compare total-to-total and compute-to-compute."
        )
    content.append("Kernel naming: canonical family/path is shown first, raw microbench label is kept in brackets.")
    if summary:
        content.append("")
        content.extend(status_lines(title, summary))
        content.append("")
        content.append("Summary (best kernel per backend)")
        content.append(format_summary_table(summary))
        content.append("")
    content.append("Full listing")
    content.append(format_table(rows))
    path.write_text("\n".join(content) + "\n")


def status_lines(title: str, rows: List[SummaryRow]) -> List[str]:
    lines: List[str] = ["Status"]
    valid_rows = [row for row in rows if row.our_ms is not None]
    acl_total_gaps = sum(1 for row in valid_rows if row.acl_ms is not None and row.acl_ms < row.our_ms)
    if title.startswith("CONV"):
        kleidiai_kernel_gaps = sum(
            1
            for row in valid_rows
            if row.kleidiai_compute_ms is not None
            and row.our_compute_ms is not None
            and row.kleidiai_compute_ms < row.our_compute_ms
        )
        kleidiai_total_gaps = sum(
            1 for row in valid_rows if row.kleidiai_total_ms is not None and row.kleidiai_total_ms < row.our_ms
        )
        kleidiai_pack_bound = sum(
            1 for row in valid_rows if row.gap_note == "pack/materialization-bound vs KleidiAI"
        )
        lines.append(f"- ACL total gaps: {acl_total_gaps}")
        lines.append(f"- KleidiAI total gaps: {kleidiai_total_gaps}")
        lines.append(f"- KleidiAI kernel-only gaps: {kleidiai_kernel_gaps}")
        lines.append(f"- KleidiAI pack-bound total gaps: {kleidiai_pack_bound}")
    else:
        kleidiai_total_gaps = sum(
            1 for row in valid_rows if row.kleidiai_total_ms is not None and row.kleidiai_total_ms < row.our_ms
        )
        lines.append(f"- ACL total gaps: {acl_total_gaps}")
        lines.append(f"- KleidiAI total gaps: {kleidiai_total_gaps}")
    return lines


def best_row(rows: List[ResultRow], prefix: str, *, include_pack_only: bool = False) -> Optional[ResultRow]:
    candidates = [row for row in rows if row.kernel.startswith(prefix) and row.ms is not None]
    if not candidates:
        return None
    if not include_pack_only:
        # Exclude auxiliary pack-only measurements from "best kernel" selection.
        candidates = [row for row in candidates if not row.kernel.endswith("_pack_only")]
        if not candidates:
            return None
    return min(candidates, key=lambda row: row.ms)  # type: ignore[arg-type]


def best_named_row(rows: List[ResultRow], kernel: str) -> Optional[ResultRow]:
    candidates = [row for row in rows if row.kernel == kernel and row.ms is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda row: row.ms)  # type: ignore[arg-type]


def summarize_rows(
    rows: List[ResultRow],
    shapes: List[ShapeSpec],
    src_order: List[str],
    pack_only_ms: Optional[dict[tuple[str, str], float]] = None,
) -> List[SummaryRow]:
    by_key: dict[tuple[str, str], List[ResultRow]] = {}
    for row in rows:
        by_key.setdefault((row.shape, row.src), []).append(row)

    summary: List[SummaryRow] = []
    for shape in shapes:
        for src in src_order:
            group = by_key.get((shape.label, src), [])
            our = best_row(group, "our_")
            acl = best_row(group, "acl_")
            kleidiai_total = None
            kleidiai_compute = None
            if shape.label.startswith("M"):
                kleidiai_total = best_named_row(group, "kleidiai_gemm_total")
                kleidiai_compute = best_named_row(group, "kleidiai_gemm")
            elif shape.label.find("K1x1") != -1:
                kleidiai_total = best_named_row(group, "kleidiai_conv1x1_total")
                kleidiai_compute = best_named_row(group, "kleidiai_conv1x1")
            elif shape.label.find("K3x3") != -1:
                kleidiai_total = best_named_row(group, "kleidiai_conv3x3_total")
                kleidiai_compute = best_named_row(group, "kleidiai_conv3x3")
            elif shape.label.find("K5x5") != -1:
                kleidiai_total = best_named_row(group, "kleidiai_conv5x5_total")
                kleidiai_compute = best_named_row(group, "kleidiai_conv5x5")
            else:
                kleidiai_total = best_row(group, "kleidiai_")
                kleidiai_compute = kleidiai_total
            our_ms = our.ms if our else None
            our_pack = pack_only_ms.get((shape.label, src)) if pack_only_ms else None
            our_compute = (our_ms - our_pack) if (our_ms is not None and our_pack is not None) else None
            acl_ms = acl.ms if acl else None
            kleidiai_total_ms = kleidiai_total.ms if kleidiai_total else None
            kleidiai_compute_ms = kleidiai_compute.ms if kleidiai_compute else None
            kleidiai_compute_speedup = (
                (kleidiai_compute_ms / our_compute) if (kleidiai_compute_ms and our_compute) else None
            )
            gap_note = ""
            if kleidiai_total_ms is not None and our_ms is not None and our_compute is not None:
                if kleidiai_compute_ms is not None and kleidiai_compute_ms < our_compute:
                    gap_note = "compute-bound vs KleidiAI"
                elif kleidiai_total_ms < our_ms:
                    gap_note = "pack/materialization-bound vs KleidiAI"
                else:
                    gap_note = "ahead of KleidiAI"
            summary.append(
                SummaryRow(
                    shape=shape.label,
                    src=src,
                    our_kernel=our.kernel if our else "",
                    our_ms=our_ms,
                    our_pack_ms=our_pack,
                    our_compute_ms=our_compute,
                    acl_ms=acl_ms,
                    kleidiai_total_ms=kleidiai_total_ms,
                    kleidiai_compute_ms=kleidiai_compute_ms,
                    speedup_acl=(acl_ms / our_ms) if (acl_ms and our_ms) else None,
                    speedup_kleidiai_total=(kleidiai_total_ms / our_ms) if (kleidiai_total_ms and our_ms) else None,
                    speedup_kleidiai_compute=kleidiai_compute_speedup,
                    gap_note=gap_note,
                )
            )
    return summary


def write_overview(path: Path, gemm_summary: Optional[List[SummaryRow]], conv_summary: Optional[List[SummaryRow]]) -> None:
    lines: List[str] = ["INT8 microbench overview", ""]
    if gemm_summary:
        lines.append("BRGEMM status")
        lines.extend(status_lines("BRGEMM", gemm_summary))
        lines.append("")
    if conv_summary:
        lines.append("CONV status")
        lines.extend(status_lines("CONV", conv_summary))
        lines.append("")
    if gemm_summary:
        lines.append("BRGEMM summary")
        lines.append(format_summary_table(gemm_summary))
        lines.append("")
    if conv_summary:
        lines.append("CONV summary")
        lines.append(format_summary_table(conv_summary))
        lines.append("")
    path.write_text("\n".join(lines))


def parse_ms(text: str) -> Optional[float]:
    return None if text == "n/a" else float(text)


def parse_speedup(text: str) -> Optional[float]:
    return None if text == "n/a" else float(text.rstrip("x"))


def load_summary_from_table(path: Path) -> Optional[List[SummaryRow]]:
    if not path.exists():
        return None

    lines = path.read_text().splitlines()
    try:
        start = lines.index("Summary (best kernel per backend)") + 1
    except ValueError:
        return None

    table_lines: List[str] = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    if len(table_lines) < 3:
        return None

    rows: List[SummaryRow] = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 13:
            continue
        rows.append(
            SummaryRow(
                shape=cells[0],
                src=cells[1],
                our_kernel=cells[2] if cells[2] != "n/a" else "",
                our_ms=parse_ms(cells[3]),
                our_pack_ms=parse_ms(cells[4]),
                our_compute_ms=parse_ms(cells[5]),
                acl_ms=parse_ms(cells[6]),
                kleidiai_total_ms=parse_ms(cells[7]),
                kleidiai_compute_ms=parse_ms(cells[8]),
                speedup_acl=parse_speedup(cells[9]),
                speedup_kleidiai_total=parse_speedup(cells[10]),
                speedup_kleidiai_compute=parse_speedup(cells[11]),
                gap_note=cells[12],
            )
        )
    return rows or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect INT8 microbench performance tables.")
    parser.add_argument("--bin", type=Path, default=default_bin_path(), help="Path to ov_cpu_int8_microbench")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent, help="Output directory")
    parser.add_argument("--signed-only", action="store_true", help="Measure only signed src (i8)")
    parser.add_argument("--unsigned-only", action="store_true", help="Measure only unsigned src (u8)")
    parser.add_argument("--gemm-only", action="store_true", help="Collect only GEMM table")
    parser.add_argument("--conv-only", action="store_true", help="Collect only CONV table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bin_path = args.bin
    if not bin_path.exists():
        raise SystemExit(f"microbench not found: {bin_path}")

    if args.signed_only and args.unsigned_only:
        raise SystemExit("choose only one of --signed-only or --unsigned-only")

    signed_modes = [True, False]
    if args.signed_only:
        signed_modes = [True]
    if args.unsigned_only:
        signed_modes = [False]

    gemm_rows: List[ResultRow] = []
    conv_rows: List[ResultRow] = []

    if not args.conv_only:
        for signed in signed_modes:
            src_label = "s8" if signed else "u8"
            for shape in default_gemm_shapes():
                output = run_microbench(bin_path, "gemm", shape.args, signed)
                gemm_rows.extend(parse_output("GEMM", shape.label, src_label, output))

    if not args.gemm_only:
        conv_pack_only: dict[tuple[str, str], float] = {}
        for signed in signed_modes:
            src_label = "s8" if signed else "u8"
            for shape in default_conv_shapes():
                output = run_microbench(bin_path, "conv", shape.args, signed)
                conv_rows.extend(parse_output("CONV", shape.label, src_label, output))
                pack_out = run_microbench(bin_path, "conv", list(shape.args) + ["--pack-only"], signed)
                pack_rows = parse_output("CONV", shape.label, src_label, pack_out)
                # For the auxiliary pack-only run we *want* the _pack_only line (there should be exactly one).
                pack = best_row(pack_rows, "our_", include_pack_only=True)
                if pack and pack.kernel.endswith("_pack_only") and pack.ms is not None:
                    conv_pack_only[(shape.label, src_label)] = pack.ms

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gemm_summary: Optional[List[SummaryRow]] = None
    conv_summary: Optional[List[SummaryRow]] = None

    if gemm_rows and not args.conv_only:
        gemm_summary = summarize_rows(gemm_rows, default_gemm_shapes(), ["s8", "u8"])
        write_table(out_dir / "brgemm_perf_table.txt", "BRGEMM (GEMM) INT8 microbench", gemm_rows, gemm_summary)

    if conv_rows and not args.gemm_only:
        conv_summary = summarize_rows(conv_rows, default_conv_shapes(), ["s8", "u8"], conv_pack_only)
        write_table(out_dir / "conv_perf_table.txt", "CONV INT8 microbench", conv_rows, conv_summary)

    if gemm_summary is None and args.conv_only:
        gemm_summary = load_summary_from_table(out_dir / "brgemm_perf_table.txt")
    if conv_summary is None and args.gemm_only:
        conv_summary = load_summary_from_table(out_dir / "conv_perf_table.txt")

    if gemm_summary or conv_summary:
        write_overview(out_dir / "int8_perf_overview.txt", gemm_summary, conv_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
