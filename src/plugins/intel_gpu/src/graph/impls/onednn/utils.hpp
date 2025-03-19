// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
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
dnnl::memory::dims convert_gemm_dims(const std::vector<int32_t> &sizes, size_t dims, bool batched_dims_can_be_removed);
dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims = 2);
dnnl::memory::dims flatten_tensor(cldnn::tensor t);
dnnl::memory::dims get_strides(dnnl::memory::dims dims);
dnnl::memory::data_type convert_data_type(cldnn::data_types dt);
dnnl::memory::format_tag convert_data_format(cldnn::format fmt);
cldnn::format convert_data_format(dnnl::memory::format_tag fmt);
dnnl::memory::format_tag convert_gemm_data_format(dnnl::memory::dims dims, format target);
dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt = dnnl::memory::format_tag::undef, bool flatten = false);
dnnl::algorithm convert_activation_func(cldnn::activation_func func);
std::vector<std::vector<size_t>> get_candidate_orders(dnnl::memory::desc desc);
cldnn::format find_format(dnnl::memory::desc desc, bool is_grouped = false);
cldnn::format find_data_format(dnnl::memory::desc desc);
dnnl::memory::format_tag get_format_by_desc(dnnl::memory::desc desc);
cldnn::format_traits convert_memory_desc_to_traits(const dnnl::memory::desc& desc, bool is_weights = false, bool is_grouped = false);
int64_t get_offset(cldnn::layout&& l, dnnl::memory::desc&& desc);
bool keep_weights_reorder_shape_consistent(cldnn::layout& layout, const dnnl::memory::desc& desc);
size_t get_post_ops_count(const program_node& node);
bool is_supported_post_ops(const program_node& node);
bool is_supported_pad(const layout& layout);

// Check if data node is per-tensor
template <typename T>
bool is_per_tensor(cldnn::data_node& node, int32_t& zp_val);

struct WeightsReorderParamsOneDNN : public cldnn::WeightsReorderParams {
    WeightsReorderParamsOneDNN(const layout& in_layout,
                               const layout& out_layout,
                               const dnnl::memory::desc& in_desc,
                               const dnnl::memory::desc& out_desc,
                               bool transposed, bool grouped = false)
        : WeightsReorderParams(in_layout, out_layout, transposed, grouped)
        , _in_desc(in_desc)
        , _out_desc(out_desc) {}

    dnnl::memory::desc _in_desc;
    dnnl::memory::desc _out_desc;
};

}  // namespace onednn
}  // namespace cldnn
