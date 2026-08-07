// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "kv_axes_position.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Reduces the past_key_values buffer size (and, optionally, the SDPA attention mask
// width) for layers configured as Sliding Window Attention (SWA), e.g. Gemma4's
// "sliding_attention" layers. Must run AFTER ReshapeToStatic (it re-shrinks a subset
// of already-static past_key_values Parameters) and BEFORE DecomposeGQA / value-tensor
// optimization passes.
//
// Step 1 (always performed): for every past_key_values.<layer>.key/value Parameter that
// belongs to a sliding-window layer, shrink its seq_len axis from
// (kvcache_size - input_size) down to (min(window_size, kvcache_size) - input_size),
// clamped to 0. This is what actually reduces per-layer NPU memory/compute for SWA.
//
// Step 2 (only when trim_attention_mask == true): for the same sliding-window layers,
// insert a shared `Slice` on the SDPA node's attention-mask input (port 3), trimming its
// last (key) axis down to the same `min(window_size, kvcache_size)` width. This is only
// valid for models where every query position shares the same trailing K/V window - i.e.
// the *generate* model (decode step, `max_generation_token_len` new tokens). It must NOT
// be applied to the *prefill* model, where multiple query positions attend to different,
// staggered windows over the same K/V span (that per-token banded pattern is produced by
// the existing SlidingWindowMask pass and must not be blanket-trimmed here).
class PatchSlidingWindowKVCache : public ov::pass::ModelPass {
    uint32_t m_window_size;
    std::vector<bool> m_layer_is_sliding;
    uint32_t m_kvcache_size;
    uint32_t m_input_size;
    KVAxesPosition m_kv_axes_position;
    bool m_trim_attention_mask;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PatchSlidingWindowKVCache");
    PatchSlidingWindowKVCache(uint32_t window_size,
                              std::vector<bool> layer_is_sliding,
                              uint32_t kvcache_size,
                              uint32_t input_size,
                              const KVAxesPosition& kv_axes_position,
                              bool trim_attention_mask);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
