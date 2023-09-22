// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/serialization/polymorphic_serializer.hpp"
#include "intel_gpu/graph/serialization/layout_serializer.hpp"
#include "intel_gpu/graph/serialization/string_serializer.hpp"
#include "intel_gpu/graph/serialization/vector_serializer.hpp"
#include "intel_gpu/runtime/device_info.hpp"

#include <string>
#include <vector>

namespace cldnn {

size_t kernel_impl_params::hash() const {
    size_t seed = 0;
    if (desc != nullptr)
        seed = desc->hash();
    const size_t prime_number = 2654435761; // magic number to reduce hash collision rate.
    for (auto& in : input_layouts) {
        seed = hash_combine(seed, in.hash() * prime_number);
    }
    for (auto& out : output_layouts) {
        seed = hash_combine(seed, out.hash() * prime_number);
    }

    // hash for fused prims
    for (auto& fd : fused_desc) {
        seed = hash_combine(seed, fd.desc->hash());
    }

    seed = hash_combine(seed, _can_be_optimized);
    return seed;
}

bool kernel_impl_params::operator==(const kernel_impl_params& rhs) const {
    if ((desc != nullptr && rhs.desc == nullptr) || (desc == nullptr && rhs.desc != nullptr))
        return false;

    if ((desc != nullptr && rhs.desc != nullptr) && *desc != *rhs.desc)
        return false;

    if (rhs.input_layouts.size() != input_layouts.size())
        return false;

    if (rhs.output_layouts.size() != output_layouts.size())
        return false;

    for (size_t i = 0; i < input_layouts.size(); i++) {
        if (input_layouts[i] != rhs.input_layouts[i])
            return false;
    }

    for (size_t i = 0; i < output_layouts.size(); i++) {
        if (output_layouts[i] != rhs.output_layouts[i])
            return false;
    }

    if (fused_desc.size() != rhs.fused_desc.size())
        return false;

    for (size_t i = 0; i < rhs.fused_desc.size(); i++) {
        if (fused_desc[i] != rhs.fused_desc[i])
            return false;
    }

    return true;
}

void kernel_impl_params::save(BinaryOutputBuffer& ob) const {
    ob << desc;
    ob << static_cast<uint64_t>(dev_type);
    ob << has_runtime_layouts;
    ob << unique_id;
    ob << input_layouts;
    ob << output_layouts;
    ob << input_offsets.size();
    for (size_t i = 0; i < input_offsets.size(); i++) {
        ob << input_offsets[i].sizes();
    }

    if (weights_layout.has_value()) {
        ob << true;
        ob << weights_layout.value();
    } else {
        ob << false;
    }

    if (bias_layout.has_value()) {
        ob << true;
        ob << bias_layout.value();
    } else {
        ob << false;
    }

    if (weights_zero_points_layout.has_value()) {
        ob << true;
        ob << weights_zero_points_layout.value();
    } else {
        ob << false;
    }

    if (activations_zero_points_layout.has_value()) {
        ob << true;
        ob << activations_zero_points_layout.value();
    } else {
        ob << false;
    }

    if (compensation_layout.has_value()) {
        ob << true;
        ob << compensation_layout.value();
    } else {
        ob << false;
    }

    ob << fused_desc.size();
#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t num_fused_prims = fused_desc_onednn.size();
    ob << num_fused_prims;
    for (auto fused_prim : fused_desc_onednn) {
        ob << make_data(&fused_prim.op_type, sizeof(onednn_post_op_type));
        ob << fused_prim.mem_offset;
        ob << fused_prim.mem_dep;
        ob << make_data(&fused_prim.tag, sizeof(dnnl::memory::format_tag));
        ob << fused_prim.flatten;
        ob << fused_prim.dims;
        ob << make_data(&fused_prim.dt, sizeof(dnnl::memory::data_type));
    }
#endif // ENABLE_ONEDNN_FOR_GPU
    ob << primary_input_idx;
}

void kernel_impl_params::load(BinaryInputBuffer& ib) {
    prog = nullptr;
    ib >> desc;
    size_t dev_type_id = 0;
    ib >> dev_type_id;
    dev_type = static_cast<cldnn::device_type>(dev_type_id);
    ib >> has_runtime_layouts;
    ib >> unique_id;
    ib >> input_layouts;
    ib >> output_layouts;
    {
        size_t num_input_offsets;
        ib >> num_input_offsets;
        input_offsets.resize(num_input_offsets);
        for (size_t i = 0; i < num_input_offsets; i++) {
            std::vector<cldnn::tensor::value_type> sizes;
            ib >> sizes;
            input_offsets[i] = cldnn::tensor(sizes);
        }
    }
    bool has_value = false;
    layout layout_buf;

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        weights_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        bias_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        weights_zero_points_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        activations_zero_points_layout = layout_buf;
    }

    ib >> has_value;
    if (has_value) {
        ib >> layout_buf;
        compensation_layout = layout_buf;
    }

    {
        // Fake fused_desc just for has_fused_primitives()
        size_t num_fused_desc;
        ib >> num_fused_desc;
        if (num_fused_desc > 0) {
            fused_desc.emplace_back(cldnn::fused_primitive_desc(nullptr));
        }
    }
#ifdef ENABLE_ONEDNN_FOR_GPU
    size_t num_fused_prims;
    ib >> num_fused_prims;
    fused_desc_onednn.resize(num_fused_prims);
    for (size_t idx = 0; idx < num_fused_prims; ++idx) {
        ib >> make_data(&fused_desc_onednn[idx].op_type, sizeof(onednn_post_op_type));
        ib >> fused_desc_onednn[idx].mem_offset;
        ib >> fused_desc_onednn[idx].mem_dep;
        ib >> make_data(&fused_desc_onednn[idx].tag, sizeof(dnnl::memory::format_tag));
        ib >> fused_desc_onednn[idx].flatten;
        ib >> fused_desc_onednn[idx].dims;
        ib >> make_data(&fused_desc_onednn[idx].dt, sizeof(dnnl::memory::data_type));
    }
#endif // ENABLE_ONEDNN_FOR_GPU
    ib >> primary_input_idx;
}

}  // namespace cldnn
