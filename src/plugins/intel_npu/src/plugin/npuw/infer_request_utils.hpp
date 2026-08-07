// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <optional>
#include <string>

#include "llm_compiled_model_utils.hpp"
#include "logging.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/itensor.hpp"
#include "openvino/runtime/so_ptr.hpp"

namespace ov {
namespace npuw {
namespace util {

ov::SoPtr<ov::ITensor> make_tensor_slice(ov::SoPtr<ov::ITensor> tensor,
                                         uint32_t dim,
                                         uint32_t start_pos,
                                         uint32_t end_pos);

void copy_by_planes(ov::SoPtr<ov::ITensor> src_tensor, ov::SoPtr<ov::ITensor> dst_tensor);

void copy_columns_by_row_chunks(ov::SoPtr<ov::ITensor> src, ov::SoPtr<ov::ITensor>& dst);

void copy_to_right(const ov::SoPtr<ov::ITensor>& src, const ov::SoPtr<ov::ITensor>& dst);

void copy_tensor_by_dim(ov::SoPtr<ov::ITensor> src_tensor,
                        ov::SoPtr<ov::ITensor> dst_tensor,
                        uint32_t kv_dim_src,
                        uint32_t kv_dim_dst);

// Physical layout maintained by write_kv_slice_sliding() for a saturated (capacity-C,
// total_tokens > C) sliding-window buffer:
//   - LeftAligned: the most recent C tokens are kept left-aligned at [0, C), in strict
//     chronological order. Reaching this state requires shifting the surviving tail
//     forward every time the window (re)saturates.
//   - Circular: token at absolute position p always lives at physical index (p % C) -
//     no data is ever moved, only overwritten in place. This is mathematically
//     equivalent for callers whose only consumers are (a) the compiled model's mask,
//     which gates the past-KV region with an unconditional "structural index <
//     past_kv_len" check (no per-column temporal identity check - see
//     rebuild_sliding_window_mask() in sliding_window_mask.cpp) and (b) attention
//     itself, which is a permutation-invariant reduction over the visible K/V columns.
//     RoPE is baked into K once at write time (using the token's own true absolute
//     position), never recomputed from buffer physical position at read time, so
//     reordering physical slots doesn't affect correctness either.
//     Circular MUST NOT be used for a buffer that is ever read back elsewhere as a
//     plain contiguous/left-aligned source (e.g. Continuous strategy's variant-switch
//     migration or chunked-prefill past-KV reuse) unless that reader is updated to
//     unwrap the circular layout first.
enum class SlidingBufferLayout { LeftAligned, Circular };

// Writes `src_new_kv` (holding the freshly-produced KV content, of which the last
// `num_new_tokens` entries along `src_kv_dim` are meaningful) into `dst_tensor`'s
// past-KV buffer along `dst_kv_dim`, honoring `dst_tensor`'s own capacity
// (dst_tensor->get_shape()[dst_kv_dim]) - which may be smaller than the logical
// "total tokens seen so far" for Sliding Window Attention (SWA) layers, since their
// past_key_values Parameter is reshaped to the (smaller) window size at compile time.
//
// `layout` selects the physical arrangement used once the window (re)saturates - see
// SlidingBufferLayout above. Defaults to LeftAligned, matching all pre-existing
// callers' expectations.
//
// For non-SWA layers (capacity >= total tokens ever seen), this is equivalent to the
// original unconditional "write at [old_total, new_total)" behavior in both layouts -
// no shifting/wrapping ever occurs and this call is a drop-in replacement.
//
// NB: `src_new_kv` is expected to follow the "present/output" convention - its
// meaningful content is right-aligned at the tail (mirrors update_kvcache_for's
// existing src_seq_len > num_tokens handling). Persistent *past* buffers reused as a
// source (e.g. chunked-prefill's own past_key_values) are LEFT-aligned instead and
// must be pre-sliced by the caller to their valid prefix before being passed in here.
void write_kv_slice_sliding(ov::SoPtr<ov::ITensor> dst_tensor,
                            ov::SoPtr<ov::ITensor> src_new_kv,
                            uint32_t dst_kv_dim,
                            uint32_t src_kv_dim,
                            uint32_t num_stored_tokens_before,
                            uint32_t num_new_tokens,
                            SlidingBufferLayout layout = SlidingBufferLayout::LeftAligned);

std::optional<ov::Output<const ov::Node>> find_port_by_name(const std::vector<ov::Output<const ov::Node>>& ports,
                                                            const std::string& name);
/**
 * @brief Searches for a port within a collection that matches any of the specified names.
 */
std::optional<ov::Output<const ov::Node>> find_port_by_names(const std::vector<ov::Output<const ov::Node>>& ports,
                                                             const std::unordered_set<std::string>& names);

void pad_position_ids(const ov::SoPtr<ov::ITensor>& padded_position_ids, const ov::SoPtr<ov::ITensor>& position_ids);

// Copy chunk_tokens from src starting at src_offset_tokens into dst, right-aligned on seq_len dim.
// Leading bytes in dst are left unchanged.
void copy_per_layer_inputs_chunk_to_right(const ov::SoPtr<ov::ITensor>& src,
                                          const ov::SoPtr<ov::ITensor>& dst,
                                          uint32_t src_offset_tokens,
                                          uint32_t chunk_tokens);

}  // namespace util
}  // namespace npuw
}  // namespace ov
