// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
class IAsyncInferRequest;
class ITensor;
class Model;
}  // namespace ov

namespace ov {
namespace npuw {
namespace util {

// Hybrid SWA layout descriptor.
// window_size == 0 means SWA is disabled.
struct SwaLayout {
    uint32_t window_size = 0;            // 0 == Sliding Window Attention disabled
    std::vector<bool> layer_is_sliding;  // per-layer flag, indexed by decoder layer id

    bool enabled() const {
        return window_size > 0;
    }

    // True if SWA is enabled and layer_idx is configured as a sliding-window layer.
    bool is_sliding(size_t layer_idx) const {
        return enabled() && layer_idx < layer_is_sliding.size() && layer_is_sliding[layer_idx];
    }
};

// rt_info key stamped on SWA-managed past_key_values Parameters whose seq_len
// axis was shrunk to the SWA window size.
static constexpr const char* NPUW_KV_CACHE_SLIDING_RT_KEY = "npuw_kv_cache_sliding";

// Derives hybrid SWA layout from per-layer SDPA mask annotations.
// Enables SWA only for genuine hybrid models: at least one sliding layer and one
// full/causal layer. Throws if sliding layers use different window sizes.
SwaLayout detect_swa_layout(const std::shared_ptr<ov::Model>& model);

// Fills additive SWA causal mask tensor (f32):
//   0.0f for visible positions, -inf (fp16 lowest cast to f32) for masked ones.
// Mask shape is [..., row_dim, col_dim], where past_width = col_dim - row_dim.
// Past slots are interpreted using circular storage (slot = abs_pos % past_width).
// Example (saturated): past_width=4, P=6 -> r=2, slot->abs = [4,5,2,3].
void fill_causal_sliding_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                              uint32_t num_stored_tokens_before,
                              uint32_t num_real_new_tokens,
                              uint32_t window_size);

// Overlays bidirectional visibility for same-image vision tokens.
// token_type_ids_real is per-call right-aligned token metadata:
//   0 = text, 1 = vision.
// Each contiguous run of 1s is treated as one image group.
// This only adds visibility within the current chunk diagonal block.
void overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                       const int64_t* token_type_ids_real,
                                       uint32_t num_real_new_tokens);

// Fills sliding_window_attention_mask input when present.
// No-op for non-hybrid models that do not expose this input.
void fill_attention_masks(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                          const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                          uint32_t num_stored_tokens_before,
                          uint32_t num_real_new_tokens,
                          uint32_t window_size,
                          const int64_t* token_type_ids_real = nullptr);

// Writes SWA KV deltas into a left-aligned past buffer.
// If the window is saturated, shifts the surviving tail to the front before append.
void write_swa_kv_slice_left_aligned(ov::SoPtr<ov::ITensor> dst_tensor,
                                     ov::SoPtr<ov::ITensor> src_new_kv,
                                     uint32_t dst_kv_dim,
                                     uint32_t src_kv_dim,
                                     uint32_t num_stored_tokens_before,
                                     uint32_t num_new_tokens);

// Writes SWA KV deltas into a circular past buffer.
// Token at absolute position p is written to physical slot (p % capacity).
void write_swa_kv_slice_circular(ov::SoPtr<ov::ITensor> dst_tensor,
                                 ov::SoPtr<ov::ITensor> src_new_kv,
                                 uint32_t dst_kv_dim,
                                 uint32_t src_kv_dim,
                                 uint32_t num_stored_tokens_before,
                                 uint32_t num_new_tokens);

}  // namespace util
}  // namespace npuw
}  // namespace ov
