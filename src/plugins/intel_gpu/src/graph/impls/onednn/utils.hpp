// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include <intel_gpu/runtime/layout.hpp>
#include <intel_gpu/runtime/engine.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/primitives/activation.hpp>
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
dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims = 2);
dnnl::memory::dims flatten_tensor(cldnn::tensor t);
dnnl::memory::data_type convert_data_type(cldnn::data_types dt);
dnnl::memory::format_tag convert_data_format(cldnn::format fmt);
dnnl::memory::format_tag convert_gemm_data_format(dnnl::memory::dims dims);
dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt = dnnl::memory::format_tag::undef, bool flatten = false);
dnnl::algorithm convert_activation_func(cldnn::activation_func func);
cldnn::format find_format(dnnl::memory::desc desc, bool is_grouped = false);

int64_t get_f_offset(cldnn::layout l, dnnl::memory::desc desc);

// If the values in the tensor are identical, make it as per-tensor value
template <typename T>
void make_per_tensor_if_possible(cldnn::data_node& node);

}  // namespace onednn
}  // namespace cldnn
