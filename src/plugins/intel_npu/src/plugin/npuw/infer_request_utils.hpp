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

}  // namespace util
}  // namespace npuw
}  // namespace ov
