// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Centralized string constants for:
//   1. canonical GGUF *file* metadata keys (as stored in the .gguf header), and
//   2. the OpenVINO rt-info schema (SPEC 3.3) under the frozen top segment "gguf".
//
// Keep this file as the single source of truth for both the on-disk keys we read and the
// rt-info path segments we write, so the producer (frontend) and consumers (genai) agree.

#pragma once

#include <string>

namespace ov {
namespace frontend {
namespace gguf {

// --- GGUF file metadata keys (architecture-independent) ---
namespace file_keys {
inline constexpr const char* general_architecture = "general.architecture";
inline constexpr const char* general_name = "general.name";
inline constexpr const char* general_file_type = "general.file_type";
inline constexpr const char* general_alignment = "general.alignment";

inline constexpr const char* tokenizer_model = "tokenizer.ggml.model";
inline constexpr const char* tokenizer_pre = "tokenizer.ggml.pre";
inline constexpr const char* tokenizer_tokens = "tokenizer.ggml.tokens";
inline constexpr const char* tokenizer_scores = "tokenizer.ggml.scores";
inline constexpr const char* tokenizer_token_type = "tokenizer.ggml.token_type";
inline constexpr const char* tokenizer_merges = "tokenizer.ggml.merges";
inline constexpr const char* tokenizer_bos_id = "tokenizer.ggml.bos_token_id";
inline constexpr const char* tokenizer_eos_id = "tokenizer.ggml.eos_token_id";
inline constexpr const char* tokenizer_chat_template = "tokenizer.chat_template";
}  // namespace file_keys

// Architecture-prefixed GGUF metadata keys, e.g. arch_key("qwen3", "context_length").
inline std::string arch_key(const std::string& arch, const std::string& suffix) {
    return arch + "." + suffix;
}

// Architecture-specific GGUF key suffixes (combined with the architecture prefix).
namespace arch_suffix {
inline constexpr const char* context_length = "context_length";
inline constexpr const char* embedding_length = "embedding_length";
inline constexpr const char* block_count = "block_count";
inline constexpr const char* feed_forward_length = "feed_forward_length";
inline constexpr const char* attention_head_count = "attention.head_count";
inline constexpr const char* attention_head_count_kv = "attention.head_count_kv";
inline constexpr const char* attention_layer_norm_rms_eps = "attention.layer_norm_rms_epsilon";
inline constexpr const char* attention_key_length = "attention.key_length";
inline constexpr const char* attention_value_length = "attention.value_length";
inline constexpr const char* rope_dimension_count = "rope.dimension_count";
inline constexpr const char* rope_freq_base = "rope.freq_base";
inline constexpr const char* rope_scaling_type = "rope.scaling.type";
inline constexpr const char* rope_scaling_factor = "rope.scaling.factor";
}  // namespace arch_suffix

// --- OpenVINO rt-info schema (SPEC 3.3) ---
// All entries live under the frozen top path segment "gguf". Each constant below is the
// remaining sub-path written with ov::Model::set_rt_info(value, rt_top, <segments...>).
namespace rt_info {
inline constexpr const char* top = "gguf";
}  // namespace rt_info

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
