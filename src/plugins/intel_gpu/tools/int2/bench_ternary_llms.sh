#!/bin/bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Batch-1 greedy decode throughput of ternary (u2) LLM IRs on one GPU, as reported in the BITCOS
# paper (Fig. 14, int2 bars): <tokens> generated tokens / (prefill + generation), EOS suppressed,
# mean over reps 2..N (rep 1 warms the OpenCL program cache).
#
#   bench_ternary_llms.sh <model> [<model> ...]
#     <model>: an IR directory (openvino_model.xml). Qwen3-family models (Bonsai 1.7B-8B, CAT-Q) run on
#              paged_bench_llm; 27B-class models need a separate embeddings IR and run on paged_bench_llm_27b,
#              given as <dir>:<embeddings.xml> or found as <dir>/openvino_text_embeddings_model.xml.
#   env: TOOLS (build dir of this folder, default ./build), TOKENS (256), REPS (3)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TOOLS=${TOOLS:-$HERE/build}
TOKENS=${TOKENS:-256}
REPS=${REPS:-3}
export BENCH_PRECISION=f16 BENCH_NO_EOS=1 BENCH_MAX_LEN=512
# Qwen3 chat template around "Tell me about photosynthesis in 200 words"; the 27B prompt is its own file.
QWEN3_IDS=151644,872,198,840,20772,7249,73667,304,1378,22870,13,151645,198,151644,77091,198,151667,271,151668,271
IDS27=$(cat "$HERE/prompt_photosynthesis_27b.txt")

printf '%-24s %10s %10s %12s\n' model "TTFT(ms)" "decode t/s" "e2e t/s"
for spec in "$@"; do
    dir=${spec%%:*}
    emb=${spec#*:}
    [[ $emb == "$spec" ]] && emb=$dir/openvino_text_embeddings_model.xml
    ttft=0 dec=0 e2e=0 n=0
    for r in $(seq "$REPS"); do
        if [[ -f $emb ]]; then
            out=$("$TOOLS/paged_bench_llm_27b" "$dir/openvino_model.xml" "$emb" GPU "$TOKENS" "$IDS27")
        else
            out=$("$TOOLS/paged_bench_llm" "$dir/openvino_model.xml" GPU "$TOKENS" "$QWEN3_IDS")
        fi
        [[ $r -eq 1 && $REPS -gt 1 ]] && continue
        t=$(awk '/^TTFT/ {print $4}' <<<"$out")
        s=$(awk '/^decode/ {print $6}' <<<"$out")
        read -r ttft dec e2e < <(awk -v a="$ttft" -v b="$dec" -v c="$e2e" -v t="$t" -v s="$s" -v k="$TOKENS" \
            'BEGIN {print a + t, b + (k - 1) / s, c + k / (t / 1000 + s)}')
        n=$((n + 1))
    done
    awk -v m="$(basename "$dir")" -v a="$ttft" -v b="$dec" -v c="$e2e" -v n="$n" \
        'BEGIN {printf "%-24s %10.1f %10.1f %12.1f\n", m, a / n, b / n, c / n}'
done
