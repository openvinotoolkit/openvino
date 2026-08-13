# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Builds and runs gen_ggml_reference.c (which links real ggml from llama.cpp) and converts
# its raw .bin dumps into the .npy files the C++ op tests load. Unlike a numpy reimplementation
# of each op's formula -- which can only confirm whatever formula the author guessed -- this
# produces *authoritative ggml outputs*, the exact values llama.cpp computes, so a translator
# that picks the wrong formula variant is caught.
#
# IMPORTANT (learned the hard way): ggml computes GELU/GELU_QUICK via an f16 lookup table
# (GGML_GELU_FP16 in ggml-cpu/vec.h): the input is rounded to f16, indexed into a 65536-entry
# f16 table, output in f16. That quantization is ~2e-3 -- LARGER than the erf-vs-tanh formula
# difference (~4e-4). So:
#   * the reference values here carry ggml's f16-table quantization;
#   * the matching C++ test tolerance must be ~3e-3, not 1e-5;
#   * at that tolerance an erf/tanh swap in the translator is NOT distinguishable for GELU
#     alone -- the real defense against formula bugs is the per-layer end-to-end diff vs
#     llama-eval-callback on a deep model (see SKILL.md "Finding the bug"), where the
#     per-call error compounds. Use these op-level vectors to catch GROSS errors (wrong op,
#     wrong axis, wrong constant), and the end-to-end diff to catch subtle formula drift.
#
# Usage:
#   python3 gen_ggml_reference.py --llama /home/vmaxim/llama.cpp --out-dir test_data

import argparse
import os
import subprocess
import sys

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama", default="/home/vmaxim/llama.cpp",
                    help="llama.cpp checkout with ggml headers and a built build-ref/bin")
    ap.add_argument("--out-dir", default="test_data")
    ap.add_argument("--build-dir", default="build-ref/bin",
                    help="path under --llama holding libggml*.so")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "gen_ggml_reference.c")
    inc = os.path.join(args.llama, "ggml", "include")
    libdir = os.path.join(args.llama, args.build_dir)
    exe = "/tmp/gen_ggml_reference"
    raw = "/tmp/ggml_ref_raw"
    os.makedirs(raw, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    cc = subprocess.run(
        ["cc", src, "-I", inc, "-L", libdir, "-lggml", "-lggml-base", "-lggml-cpu", "-lm",
         "-o", exe],
        capture_output=True, text=True)
    if cc.returncode != 0:
        print(cc.stderr, file=sys.stderr)
        sys.exit("compile failed")

    env = dict(os.environ, LD_LIBRARY_PATH=libdir)
    run = subprocess.run([exe, raw], env=env, capture_output=True, text=True)
    print(run.stdout)
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr)
        sys.exit("run failed")

    # Convert every <name>.bin + <name>.shape pair to <name>.npy.
    for fn in sorted(os.listdir(raw)):
        if not fn.endswith(".bin"):
            continue
        base = fn[:-4]
        with open(os.path.join(raw, base + ".shape")) as f:
            shape = tuple(int(x) for x in f.read().split())
        # *_qbytes are raw quantized blocks (u8, 1-D byte count); everything else is f32.
        if base.endswith("_qbytes"):
            arr = np.fromfile(os.path.join(raw, fn), dtype=np.uint8).reshape(shape)
        else:
            arr = np.fromfile(os.path.join(raw, fn), dtype=np.float32).reshape(shape)
        np.save(os.path.join(args.out_dir, base + ".npy"), arr)
        print(f"  wrote {base}.npy {arr.shape} {arr.dtype}")


if __name__ == "__main__":
    main()
