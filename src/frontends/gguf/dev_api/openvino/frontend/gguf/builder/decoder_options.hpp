// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

enum class RopeMode { Normal, Neox, Interleaved };

// Architecture overrides; unset options retain detection. Dimensions and derived plans stay internal.
struct GGUF_FRONTEND_API DecoderOptions {
    std::optional<bool> qk_norm_after_rope;
    std::optional<bool> post_norm_only;
    std::optional<bool> normalize_expert_weights;
    std::optional<bool> geglu;
    std::optional<bool> value_norm;
    std::optional<bool> embedding_norm;
    std::optional<bool> rope_on_swa_only;
    std::optional<float> post_norm_epsilon;
    std::optional<float> swa_rope_frequency_base;
    std::optional<int> swa_rope_dimensions;
    std::optional<int> sliding_window;
    std::optional<int> rope_skip_period;
};

// Value snapshot; changes do not affect the shared decoder configuration.
struct GGUF_FRONTEND_API DecoderDimensions {
    int layers;
    int embedding;
    float norm_epsilon;
};

// Per-layer snapshot from decoder_attention configuration, including KV heads and SWA RoPE.
struct GGUF_FRONTEND_API DecoderLayerParameters {
    int query_heads;
    int kv_heads;
    int head_size;
    float attention_scale;
    RopeConfig rope;
    bool sliding_window;
    bool recurrent;
};

}  // namespace ov::frontend::gguf
