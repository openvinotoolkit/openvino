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

// Fills the per-SDPA additive `sliding_window_attention_mask` tensor (f32, shape
// [..., row_dim, col_dim] with all leading dims == 1) with real sliding-window causal values:
// 0.0f where the query row may attend to a column, std::numeric_limits<float>::lowest() where it
// may not.
//
// The tensor's last two dims are read directly off its own runtime shape:
//   - row_dim: this call's query length (`input_size` for the compiled variant).
//   - col_dim: full key context width - `col_dim - row_dim` ("past_width") is the number of
//     "past" columns preceding this call's own current-chunk columns.
// The past K/V buffer has already been physically shrunk (PatchSlidingWindowKVCache) to
// `window_size` columns, always holding the most recent `min(P, window_size)` tokens
// left-aligned-by-recency (see write_kv_slice_sliding()'s invariant) - structurally
// `past_width == window_size`.
//
// `P` (num_stored_tokens before this call) and `L` (number of REAL, non-pad new tokens this call,
// right-aligned within the last `row_dim`/`col_dim` current-chunk positions - `L <= row_dim`) are
// the only per-call-site inputs needed; every other quantity is derived from them plus the
// tensor's own shape. See the NPUW SWA Phase 5 design (session plan) for the full derivation.
void fill_causal_sliding_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                              uint32_t num_stored_tokens_before,
                              uint32_t num_real_new_tokens,
                              uint32_t window_size);

// Overlays extra bidirectional visibility for same-image vision tokens on top of an
// already-filled (via fill_causal_sliding_mask()) mask tensor, mirroring the original HF-exported
// graph's vision-block "same_group" bidirectional masking (see remove_token_type_ids.cpp).
// `token_type_ids_real` must point to exactly `num_real_new_tokens` int64 values (0 = text token,
// 1 = vision token) for THIS call's real, right-aligned tokens, in order. A "group" is one maximal
// contiguous run of `1`s (one image); only forces attend=true between two vision tokens within
// the SAME run (never across separate images, never for text tokens) - only within this call's
// own current-chunk diagonal block (an image spanning a chunk/call boundary is out of scope, see
// the session plan). Never removes an already-true causal/window attend, only adds visibility.
// No-op if `num_real_new_tokens == 0`.
void overlay_vision_bidirectional_mask(ov::SoPtr<ov::ITensor> mask_tensor,
                                      const int64_t* token_type_ids_real,
                                      uint32_t num_real_new_tokens);

}  // namespace util
}  // namespace npuw
}  // namespace ov
