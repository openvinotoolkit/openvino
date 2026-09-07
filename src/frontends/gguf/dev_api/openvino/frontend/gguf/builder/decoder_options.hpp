// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>

#include "openvino/frontend/gguf/decoder.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

enum class RopeMode { Normal, Neox, Interleaved };

// Architecture facts that cannot reliably be inferred from tensor names. Unset options retain
// detection. Dimensions and derived execution plans are deliberately not writable by extensions.
struct GGUF_FRONTEND_API DecoderOptions {
    std::optional<bool> geglu;
    std::optional<bool> value_norm;
    std::optional<bool> embedding_norm;
    std::optional<bool> rope_on_swa_only;
    std::optional<float> post_norm_epsilon;
    std::optional<float> swa_rope_frequency_base;
    std::optional<int> swa_rope_dimensions;
    std::optional<int> sliding_window;
};

// Resolved dimensions for custom decoder topology. Returned by value; changing the snapshot
// does not modify the configuration used by the shared decoder blocks.
struct GGUF_FRONTEND_API DecoderDimensions {
    int layers;
    int embedding;
    float norm_epsilon;
};

// Per-layer snapshot for custom attention implementations. Values come from the same resolved
// configuration as decoder_attention, including per-layer KV heads and SWA RoPE parameters.
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
