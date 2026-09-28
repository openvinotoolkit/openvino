#!/bin/bash
# lm_eval (GSM8K by default) on the OpenVINO TernOCL int2 path, Bonsai 2 27B.
#   bash run_lm_eval_ov.sh <model-dir> <tokenizer-dir> <out-dir> [extra lm_eval_ov.py args...]
# <model-dir> holds openvino_model.xml and openvino_text_embeddings_model.xml.
# Env: LIMIT, BATCH (8), THINK (medium), TASKS, PY (python with lm_eval + transformers),
#      SERVE (paged_serve_llm_27b). Run it in the environment of the guide (env.sh).
set -uo pipefail
MD=${1:?model dir}; TOK=${2:?tokenizer dir}; OUT=${3:?output dir}; shift 3
TOOLS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY=${PY:-python3}
SERVE=${SERVE:-$TOOLS/build/paged_serve_llm_27b}
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
export OV_TERNOCL_INT2_MERGE_MLP=${OV_TERNOCL_INT2_MERGE_MLP:-1}
cd "$OUT" && "$PY" "$TOOLS/lm_eval_ov.py" --lm "$MD/openvino_model.xml" --embed "$MD/openvino_text_embeddings_model.xml" \
  --tokenizer "$TOK" --serve "$SERVE" --tasks "${TASKS:-gsm8k_cot_llama}" \
  ${LIMIT:+--limit $LIMIT} --batch "${BATCH:-8}" --think "${THINK:-medium}" --out "$OUT" "$@" 2>&1 | tee "$OUT/run.log"
