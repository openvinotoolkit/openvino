// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "../kv_cache_sliding_window_manager.hpp"
#include "kv_axes_position.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// Patch pass for genuine hybrid SWA decoder models (mixed sliding and full-attention layers).
//
// Responsibilities (single pass):
// 1) Externalize sliding-layer SDPA mask inputs to one model input:
//    "sliding_window_attention_mask".
// 2) Shrink past_key_values seq_len for sliding layers only.
// 3) Shrink externalized mask key-axis to match the same post-concat KV width.
// 4) Privatize and patch KV-length-dependent shape inputs (Broadcast/Reshape/Slice) so
//    cross-layer CSE/shared shape subgraphs cannot leak sliding sizes into full-attention layers.
//
// Sizing rule:
//   new_past =
//     0                              if input_size >= kvcache_size
//     window_size                    if window_size < kvcache_size
//     kvcache_size - input_size      otherwise
//   new_kv_total = new_past + input_size
//
// Examples:
// - generate: kvcache=192, input=1, window=32  -> new_past=32, new_kv_total=33
// - chunk prefill: kvcache=192, input=32, window=32 -> new_past=32, new_kv_total=64
//
// Must run AFTER ReshapeToStatic and BEFORE DecomposeGQA / V-tensor optimization passes.
// Runs once per model variant (prefill and each generate kv-size bucket).
class PatchSlidingWindowKVLayout : public ov::pass::ModelPass {
    ov::npuw::util::SwaLayout m_swa_layout;
    uint32_t m_kvcache_size;
    uint32_t m_input_size;
    KVAxesPosition m_kv_axes_position;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PatchSlidingWindowKVLayout");
    PatchSlidingWindowKVLayout(ov::npuw::util::SwaLayout swa_layout,
                               uint32_t kvcache_size,
                               uint32_t input_size,
                               const KVAxesPosition& kv_axes_position);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
