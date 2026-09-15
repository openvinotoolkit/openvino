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

// rt_info key stamped on SWA-managed past_key_values Parameters whose seq_len
// axis was shrunk to the SWA window size.
static constexpr const char* NPUW_KV_CACHE_SLIDING_RT_KEY = "npuw_kv_cache_sliding";

// Fills additive causal sliding-window attention mask tensor in-place:
// 0.0f for visible positions, -inf for masked ones.
void fill_causal_sliding_window_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                     uint32_t num_stored_tokens,
                                     uint32_t num_new_tokens,
                                     uint32_t window_size);

// Overlays bidirectional visibility for same image vision tokens.
// token_type_ids is per-call right-aligned token metadata (0 = text, 1 = vision)
void overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                       const int64_t* token_type_ids,
                                       uint32_t num_new_tokens);

// Fills sliding_window_attention_mask input when present.
// No-op for non-SWA models that do not expose this input.
void fill_sliding_window_attention_mask(const std::shared_ptr<ov::IAsyncInferRequest>& request,
                                        const std::unordered_map<std::string, ov::Output<const ov::Node>>& in_ports,
                                        uint32_t num_stored_tokens,
                                        uint32_t num_new_tokens,
                                        uint32_t window_size);

// Writes SWA KV deltas into a left-aligned past buffer, shifting the
// surviving tail to the front first when the window is saturated.
void write_swa_kv_slice_left_aligned(ov::SoPtr<ov::ITensor> dst_tensor,
                                     ov::SoPtr<ov::ITensor> src_new_kv,
                                     uint32_t dst_kv_dim,
                                     uint32_t src_kv_dim,
                                     uint32_t num_stored_tokens,
                                     uint32_t num_new_tokens);

// Writes SWA KV deltas into a circular past buffer
void write_swa_kv_slice_circular(ov::SoPtr<ov::ITensor> dst_tensor,
                                 ov::SoPtr<ov::ITensor> src_new_kv,
                                 uint32_t dst_kv_dim,
                                 uint32_t src_kv_dim,
                                 uint32_t num_stored_tokens,
                                 uint32_t num_new_tokens);

}  // namespace util
}  // namespace npuw
}  // namespace ov
