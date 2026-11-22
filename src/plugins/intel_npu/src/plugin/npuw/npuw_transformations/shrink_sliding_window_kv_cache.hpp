// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "kv_axes_position.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Model pass for decoders with sliding-window attention (SWA) layers, whether hybrid
// (mixed SWA and full attention layers) or uniformly SWA.
//
// Responsibilities:
// 1) Externalize the SWA SDPA mask input as one model parameter: "sliding_window_attention_mask".
// 2) Shrink the past_key_values seq_len axis for SWA layers only.
// 3) Shrink the externalized mask width to the same post-concat KV width.
// 4) Privatize and patch KV-length-dependent shape constants (Broadcast/Reshape) to prevent
//    cross-layer shared-shape leakage from SWA layers into full attention layers.
// 5) Prefix SWA past_key_values.N.key/value Parameter and present.N.key/value output names,
//    so SWA-managed KV cache entries can be identified by name alone downstream.
//
// SWA contract:
// - new_past = window_size (+ input_size - 1 slack slots when `multi_token_ring`), rounded up so that
//   input_size + new_past is a multiple of 16 (NPU HW prefers 16-aligned KV lengths).
//   The slack is for multi-token generation (speculative / tree verification): all input rows are written to the
//   ring before the pipeline keeps the accepted ones, so the rejected rows may only overwrite tokens that no later
//   row can see.
// - new_past is capped at the variant's own past capacity (kvcache_size - input_size): a KV variant smaller than the
//   window keeps its whole past (its ring never wraps), still managed and named as SWA.
// - new_kv_total = input_size + new_past.
//
// Pass ordering:
// - Run after ReshapeToStatic.
// - Run before DecomposeGQA / V-tensor optimization passes.
//
// Executed once per model variant (prefill and each generate KV-size bucket).
class ShrinkSlidingWindowKVCache : public ov::pass::ModelPass {
    uint32_t m_kvcache_size;
    uint32_t m_input_size;
    KVAxesPosition m_kv_axes_position;
    bool m_multi_token_ring;
    uint32_t m_window_size = 0;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ShrinkSlidingWindowKVCache");
    ShrinkSlidingWindowKVCache(uint32_t kvcache_size,
                               uint32_t input_size,
                               const KVAxesPosition& kv_axes_position,
                               bool multi_token_ring = false);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

    // SWA window size detected by the last run_on_model() call, 0 if none.
    uint32_t window_size() const {
        return m_window_size;
    }
};

}  // namespace ov::npuw
