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
//
// SWA contract:
// - new_past = window_size
// - new_kv_total = input_size + window_size
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

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::ShrinkSlidingWindowKVCache");
    ShrinkSlidingWindowKVCache(uint32_t kvcache_size, uint32_t input_size, const KVAxesPosition& kv_axes_position);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
