// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
class IAsyncInferRequest;
class ITensor;
}  // namespace ov

namespace ov {
namespace npuw {
namespace util {

// Fills additive SWA causal mask tensor (f32):
//   0.0f for visible positions, -inf (fp16 lowest cast to f32) for masked ones.
// Mask shape is [..., row_dim, col_dim], where past_width = col_dim - row_dim.
// Past region follows write_swa_kv_slice_circular() storage:
//   - Unsaturated (P < past_width): valid prefix [0, P), slot c maps to abs=c.
//   - Saturated   (P >= past_width, r=P%past_width): slot->abs is split by r.
// Example (saturated): past_width=4, P=6 -> r=2, slot->abs=[4,5,2,3].
// Visibility is the intersection of causal and sliding-window constraints.
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
