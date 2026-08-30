// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <set>
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

// gguf ROPE op-case (see ggml-decoder.cpp::compute_op_case). The high 16 bits encode the
// mode: NORMAL=0 (llama/minicpm: rotate consecutive pairs), NEOX=1 (qwen/phi/hunyuan:
// rotate halves). The input is not a VIEW here so the low bits stay 0.
constexpr int ROPE_OP_CASE_NORMAL = 0x00000000;
constexpr int ROPE_OP_CASE_NEOX = 0x00010000;
// IMROPE=2: interleaved multimodal rope (qwen35 / qwen3vl). inp_pos carries 4 mrope sections.
constexpr int ROPE_OP_CASE_IMROPE = 0x00020000;

// Architectures whose rope is NEOX (rotate-halves); everything else in the supported set
// uses NORMAL (rotate consecutive pairs). Mirrors llama_model_rope_type.
bool arch_uses_neox_rope(const std::string& arch);

// Architectures END-TO-END VERIFIED on the generic decoder builder: convert + compile + generation
// checked against the reference (native llama.cpp / HF) on a real checkpoint. Safe to rely on.
const std::set<std::string>& verified_archs();

// Architectures EXPECTED to work via the generic builder's GGUF-tensor-table auto-detection, but
// NOT yet end-to-end verified on a real checkpoint. Enabled (they convert), but conversion emits a
// one-time warning so callers know they are best-effort. Promote to verified_archs() once a model
// of the family has been checked against the reference. See docs/adding_an_architecture.md.
const std::set<std::string>& experimental_archs();

// All architectures the native builder accepts = verified + experimental.
const std::set<std::string>& supported_archs();

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
