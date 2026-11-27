// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct SDPARef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::ref")
    SDPARef(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
        const auto& supported_precisions = ov::intel_gpu::op::SDPA::get_supported_precisions();
        const auto is_supported_precision = [&supported_precisions](const ov::element::Type& dt) {
            return one_of(dt, supported_precisions);
        };
        // K/V inputs additionally accept quantized (i8) data, and u8 for a per-channel compressed cache (the GQA decomposition
        // stores a 4-bit one as u8 bytes of two nibbles; per-channel = the head dim is not grouped, see
        // SDPABase::has_per_channel_compressed_kv()).
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        const auto& group_sizes = desc->quantization_attributes.group_sizes;
        const bool per_channel_kv = desc->is_kv_compressed && !group_sizes.empty() && group_sizes.back() == 1;
        const auto is_supported_kv_precision = [&is_supported_precision, per_channel_kv](const ov::element::Type& dt) {
            return is_supported_precision(dt) || dt == ov::element::i8 || (per_channel_kv && dt == ov::element::u8);
        };

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!is_supported_kv_precision(k_layout.data_type) || !is_supported_kv_precision(v_layout.data_type)) {
            return false;
        }

        return is_supported_precision(q_layout.data_type) && is_supported_precision(out_layout.data_type);
    }
};

}  // namespace ov::intel_gpu::ocl
