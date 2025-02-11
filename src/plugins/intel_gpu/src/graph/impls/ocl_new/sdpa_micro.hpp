// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifdef ENABLE_ONEDNN_FOR_GPU

#include "impls/ocl/kernel_selector_helper.h"
#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "sdpa_utils.hpp"

#include <memory>

namespace ov::intel_gpu::ocl {

struct SDPAMicro : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::micro")
    SDPAMicro(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
        auto& engine = node.get_program().get_engine();
        const auto& device_info = engine.get_device_info();

        const auto supports_microkernels = cldnn::query_microkernels_supported(engine, node.get_program().get_config());
        if (device_info.arch < gpu_arch::xe_hpg || !supports_microkernels)
            return false;

        static const std::vector<ov::element::Type_t> supported_q_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };
        static const std::vector<ov::element::Type_t> supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };

        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format))
            return false;

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types))
            return false;

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types))
            return false;

        auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        if (desc->is_causal)
            return false;

        auto Q_num_heads_dim = get_num_heads(q_layout, desc->input_q_transpose_order);
        auto K_num_heads_dim = get_num_heads(k_layout, desc->input_k_transpose_order);
        auto V_num_heads_dim = get_num_heads(v_layout, desc->input_v_transpose_order);

        if (desc->input_q_transpose_order[3] != 3 || desc->input_k_transpose_order[3] != 3 || desc->input_v_transpose_order[3] != 3)
            return false;

        if (Q_num_heads_dim.is_dynamic() || K_num_heads_dim.is_dynamic() || V_num_heads_dim.is_dynamic() || K_num_heads_dim != V_num_heads_dim)
            return false;

        if (q_layout.get_partial_shape()[3].get_length() > 256)
            return false;

        auto data_inputs_num = get_data_inputs_num(*desc);

        // Do not use sdpa_micro kernel with a scalar-value mask
        if (data_inputs_num > 3 && !node.get_dependency(3).get_output_layout().is_dynamic() && node.get_dependency(3).get_output_layout().count() == 1)
            return false;

        return true;
    }

    bool support_shapes(const kernel_impl_params& param) const override {
        // TODO: Implement check for shape with indirect access required
        return true;
    }
};

}  // namespace ov::intel_gpu::ocl

#endif  // ENABLE_ONEDNN_FOR_GPU
