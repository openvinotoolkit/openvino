// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/serialization/weights_reorder_params.hpp>
#include <intel_gpu/primitives/activation.hpp>
#include "intel_gpu/primitives/reorder.hpp"
#include "intel_gpu/runtime/format.hpp"
#include <data_inst.h>

namespace cldnn {
namespace onednn {

// common utils
template <typename T>
cldnn::memory::ptr convert_zp_data_to_s32(const memory::ptr zp_memory);
cldnn::format default_fmt_for_dims(size_t dims, bool is_grouped = false);
void combine_bf_with_first_spatial_dim(cldnn::layout& l);

// cldnn -> onednn
dnnl::memory::dims convert_tensor(cldnn::tensor t, size_t dims = 2, bool is_grouped = false);
dnnl::memory::dims convert_gemm_tensor(cldnn::tensor t, size_t dims, bool batched_dims_can_be_removed);
dnnl::memory::dims convert_gemm_dims(const std::vector<ov::Dimension::value_type> &sizes, size_t dims, bool batched_dims_can_be_removed);
dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims = 2);
dnnl::memory::dims flatten_tensor(cldnn::tensor t);
dnnl::memory::dims get_strides(dnnl::memory::dims dims);
dnnl::memory::data_type convert_data_type(cldnn::data_types dt);
dnnl::memory::format_tag convert_data_format(cldnn::format fmt);
cldnn::format convert_data_format(dnnl::memory::format_tag fmt);
dnnl::memory::format_tag get_default_data_format(const cldnn::layout& l);
dnnl::memory::format_tag convert_gemm_data_format(dnnl::memory::dims dims, format target);

dnnl::memory::desc layout_to_memory_desc(
    const cldnn::layout& l,
    dnnl::memory::format_tag target_fmt = dnnl::memory::format_tag::undef);

dnnl::memory::desc layout_to_memory_desc_flatten(
    const cldnn::layout& l,
    dnnl::memory::format_tag target_fmt);

dnnl::memory::desc layout_to_memory_desc_strides(
    const cldnn::layout& l,
    dnnl::memory::format_tag target_fmt);

dnnl::memory::desc layout_to_memory_desc_blocked(
    const cldnn::layout& l,
    dnnl::memory::format_tag target_fmt);

/// This function is specifically designed for quantize post-op inputs where:
///  - For gemm/fully_connected: always use default format
///  - For other primitives: use blocked format if output is blocked, otherwise use undef
dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, bool use_default_format, bool is_output_blocked = false);

std::tuple<dnnl::memory::desc, dnnl::memory::desc, dnnl::memory::desc> get_conv_memory_descs(cldnn::layout input_layout,
                                                                 cldnn::layout weights_layout,
                                                                 cldnn::layout output_layout,
                                                                 dnnl::memory::format_tag target_fmt = dnnl::memory::format_tag::undef);
dnnl::algorithm convert_activation_func(cldnn::activation_func func);
std::vector<std::vector<size_t>> get_candidate_orders(dnnl::memory::desc desc);
cldnn::format find_format(dnnl::memory::desc desc, bool is_grouped = false);
cldnn::format find_data_format(dnnl::memory::desc desc);
dnnl::memory::format_tag get_format_by_desc(dnnl::memory::desc desc);
cldnn::format_traits convert_memory_desc_to_traits(const dnnl::memory::desc& desc, bool is_weights = false, bool is_grouped = false);
int64_t get_offset(const cldnn::layout& l, dnnl::memory::desc&& desc);
bool keep_weights_reorder_shape_consistent(cldnn::layout& layout, const dnnl::memory::desc& desc);
size_t get_post_ops_count(const program_node& node);
bool is_supported_post_ops(const program_node& node);
bool is_supported_pad(const layout& layout);

// Check if data node is per-tensor
template <typename T>
bool is_per_tensor(cldnn::data_node& node, int32_t& zp_val);

int get_prelu_mask_from_layouts(const std::function<layout()>& get_output_layout,
                                const std::function<layout(int32_t)>& get_input_layout,
                                int32_t slope_input_idx);

std::string memory_desc_to_string(const dnnl::memory::desc& desc);

}  // namespace onednn
}  // namespace cldnn
