// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <oneapi/dnnl/dnnl.hpp>

#include <cldnn/runtime/layout.hpp>
#include <cldnn/runtime/engine.hpp>
#include <cldnn/runtime/memory.hpp>
#include <cldnn/primitives/activation.hpp>

namespace cldnn {
namespace onednn {

// common utils
template <typename T>
cldnn::memory::ptr convert_zp_data_to_s32(const memory::ptr zp_memory);
cldnn::format default_fmt_for_dims(size_t dims, bool is_grouped = false);

// cldnn -> onednn
dnnl::memory::dims convert_tensor(cldnn::tensor t, size_t dims = 2, bool is_grouped = false);
dnnl::memory::dims convert_spatials(cldnn::tensor t, size_t dims = 2);
dnnl::memory::dims flatten_tensor(cldnn::tensor t);
dnnl::memory::data_type convert_data_type(cldnn::data_types dt);
dnnl::memory::format_tag convert_data_format(cldnn::format fmt);
dnnl::memory::desc layout_to_memory_desc(cldnn::layout l, dnnl::memory::format_tag target_fmt = dnnl::memory::format_tag::undef, bool flatten = false);
dnnl::memory::format_tag get_format_by_desc(dnnl::memory::desc desc);
dnnl::algorithm convert_activation_func(cldnn::activation_func func);

// onednn -> cldnn
cldnn::format convert_format(dnnl::memory::format_tag fmt, bool is_grouped = false);

}  // namespace onednn
}  // namespace cldnn
