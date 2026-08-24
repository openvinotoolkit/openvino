// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <unordered_map>

#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Which FAMILY of model a .gguf file holds. This is the first thing the builder decides, before
// any hyperparameter is read, because the metadata key layout itself differs per family: a causal
// decoder carries "<arch>.block_count" / "<arch>.attention.head_count", while an mmproj file
// carries "clip.*" keys and none of those.
//
// It is deliberately NOT the architecture. Architectures within a family (llama, qwen3, gemma4,
// ...) are data, detected from the tensor table; families are code, one ModelBuilder subclass
// each.
enum class ModelKind {
    Decoder,  // causal decoder-only LLM: the "llama family" (dense + MoE + hybrid recurrent)
    Vision,   // mmproj vision encoder + projector (clip.has_vision_encoder)
    Audio,    // mmproj audio encoder + projector (clip.has_audio_encoder)
};

// Classify a parsed GGUF file from its raw metadata.
//
// mmproj files written by llama.cpp set general.architecture = "clip" and carry
// clip.has_vision_encoder / clip.has_audio_encoder (see llama.cpp tools/mtmd/clip-impl.h
// KEY_HAS_VISION_ENC / KEY_HAS_AUDIO_ENC); every LLM GGUF names its own architecture instead.
ModelKind detect_model_kind(const std::unordered_map<std::string, GGUFMetaData>& metadata);

// Human-readable name for a kind, used in diagnostics.
const char* model_kind_name(ModelKind kind);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
