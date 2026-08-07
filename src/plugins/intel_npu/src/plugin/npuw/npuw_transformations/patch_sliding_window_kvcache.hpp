// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "kv_axes_position.hpp"
#include "openvino/pass/pass.hpp"

namespace ov::npuw {

// For a genuine hybrid Sliding-Window-Attention model (e.g. Gemma-4: some decoder layers use a
// bounded sliding window, others full/causal attention), owns the ENTIRE lifecycle of the
// externalized sliding-window attention-mask model input, plus reduces the past_key_values buffer
// size for layers configured as Sliding Window Attention (SWA), e.g. Gemma4's "sliding_attention"
// layers. Must run AFTER ReshapeToStatic (it reads already-static shapes off the about-to-be-cut
// mask subgraph, and re-shrinks already-static past_key_values Parameters) and BEFORE DecomposeGQA
// / value-tensor optimization passes. Runs once per model variant (prefill, each generate kv_size
// bucket) - each variant needs its own concretely-and-independently-shaped mask Parameter.
//
// Step 0 (always performed, first): disconnects every SlidingWindow-classified
// ScaledDotProductAttention node's mask input (classified via `DetectAttentionMask`'s rt_info, see
// NPUW_SDPA_MASK_RT_KEY) from its in-graph computed mask-construction subgraph, and reconnects it
// to a single new shared model input added here: "sliding_window_attention_mask". Its shape is
// read DIRECTLY off the real (guaranteed fully-static, since ReshapeToStatic already ran) mask
// value it replaces on the first sliding SDPA encountered - no hardcoded shape formula, single
// source of truth. Once disconnected, the original in-graph mask-computation subgraph(s) become
// unreachable and are removed by standard dead-code elimination in the pass manager - no separate
// cleanup needed.
//
// Full-attention/causal SDPAs are intentionally left completely UNTOUCHED by this step: their
// past_key_values is never resized by Step 1 below, so whatever native mask representation the
// exporter produced for them (an `is_causal=true` attribute, or an explicit mask subgraph) stays
// perfectly shape-valid on its own. Externalizing it too would require host-constructing and
// copying a full-`m_kvcache_size`-wide mask tensor on every inference call for no shape-validity
// benefit, so it is deliberately avoided.
//
// Step 1 (always performed): for every past_key_values.<layer>.key/value Parameter that
// belongs to a sliding-window layer, shrink its seq_len axis to hold exactly
// (window_size - input_size) of past history when window_size < kvcache_size (so the
// post-concat K/V total, past + input_size, equals window_size + input_size regardless
// of how input_size compares to window_size - e.g. chunked prefill with input_size ==
// window_size still needs the earliest new row's full window of history), or
// (kvcache_size - input_size), clamped to 0, when window_size >= kvcache_size (sliding
// never actually kicks in within this variant's budget - behaves like a regular layer).
// This is what actually reduces per-layer NPU memory/compute for SWA.
//
// Step 1b (always performed, for BOTH prefill and generate variants): shrink the
// `sliding_window_attention_mask` model input's last (key) axis to match the exact same
// post-concat K/V total (window_size + input_size, or kvcache_size - input_size in the
// degenerate case) as Step 1's past_key_values buffers - required for shape consistency
// at the consuming SDPA node in every variant, including prefill (Step 1's shrink is
// unconditional there too). Only the mask's CONTENT differs by variant: the generate
// model's window is uniform across the single new query row, while the prefill model's
// query rows attend to different, staggered windows over the same (now-narrower) K/V
// span - the host-side runtime is responsible for filling the content correctly for
// each case; this step only establishes the correct shape.
class PatchSlidingWindowKVCache : public ov::pass::ModelPass {
    uint32_t m_window_size;
    std::vector<bool> m_layer_is_sliding;
    uint32_t m_kvcache_size;
    uint32_t m_input_size;
    KVAxesPosition m_kv_axes_position;

public:
    OPENVINO_MODEL_PASS_RTTI("ov::npuw::PatchSlidingWindowKVCache");
    PatchSlidingWindowKVCache(uint32_t window_size,
                              std::vector<bool> layer_is_sliding,
                              uint32_t kvcache_size,
                              uint32_t input_size,
                              const KVAxesPosition& kv_axes_position);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::npuw
