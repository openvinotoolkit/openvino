#!/usr/bin/env bash
# Greedy decode of the Bonsai ternary models found under <models-dir>.
#
#   run_bonsai.sh <models-dir> [tokens]
#
# Looks for (each one is optional):
#   bonsai8b-u2/openvino_model.xml
#   bonsai27b-u2/openvino_model.xml + bonsai27b-fp16/openvino_text_embeddings_model.xml
#   bonsai2-27b-u2/openvino_model.xml + openvino_text_embeddings_model.xml
# See docs/ternocl_int2_bonsai2_from_scratch.md for how to produce them.
set -uo pipefail

MODELS=${1:?usage: run_bonsai.sh <models-dir> [tokens]}
TOKENS=${2:-256}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BENCH=$HERE/build

[ -x "$BENCH/paged_bench_llm_27b" ] || {
    echo "tools not built; see section 2 of docs/ternocl_int2_bonsai2_from_scratch.md" >&2
    exit 1
}

IDS=$(cat "$HERE/prompt_photosynthesis_27b.txt")

if [ -f "$MODELS/bonsai8b-u2/openvino_model.xml" ]; then
    echo "### 8B, paged"
    env OV_TERNOCL_INT2_MERGE_MLP=1 OV_GPU_FUSE_RMS_ROPE=1 \
        "$BENCH/paged_bench_llm" "$MODELS/bonsai8b-u2/openvino_model.xml" GPU "$TOKENS" \
        2>&1 | grep -aE "TTFT|decode |generated_ids"
fi

if [ -f "$MODELS/bonsai27b-u2/openvino_model.xml" ]; then
    echo "### 27B, paged"
    env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_TERNOCL_INT2_MERGE_MLP=1 \
        "$BENCH/paged_bench_llm_27b" "$MODELS/bonsai27b-u2/openvino_model.xml" \
        "$MODELS/bonsai27b-fp16/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
        2>&1 | grep -aE "TTFT|decode |generated_ids"

    echo "### 27B, stateful with a static decode shape"
    env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_STATIC_DECODE=1 BENCH_NO_EOS=1 \
        OV_TERNOCL_INT2_MERGE_MLP=1 \
        "$BENCH/bench_llm_27b" "$MODELS/bonsai27b-u2/openvino_model.xml" \
        "$MODELS/bonsai27b-fp16/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
        2>&1 | grep -aE "TTFT|decode |generated_ids"
fi

if [ -f "$MODELS/bonsai2-27b-u2/openvino_model.xml" ]; then
    echo "### Bonsai 2 27B, paged (fused Hadamard input transform)"
    env BENCH_PRECISION=f16 BENCH_MAX_LEN=512 BENCH_NO_EOS=1 OV_TERNOCL_INT2_MERGE_MLP=1 \
        "$BENCH/paged_bench_llm_27b" "$MODELS/bonsai2-27b-u2/openvino_model.xml" \
        "$MODELS/bonsai2-27b-u2/openvino_text_embeddings_model.xml" GPU "$TOKENS" "$IDS" \
        2>&1 | grep -aE "TTFT|decode |generated_ids"
fi
