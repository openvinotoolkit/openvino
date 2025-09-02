// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"
#include "scaled_dot_product_attention_inst.h"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {
struct SDPAStage {
    constexpr static size_t SINGLE_TOKEN = 0;
    constexpr static size_t MULTI_TOKENS = 1;
    constexpr static size_t FINALIZATION = 2;
    constexpr static size_t MICRO = 3;
};

struct SDPAOpt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::sdpa::opt")
    explicit SDPAOpt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] static bool supports_micro_sdpa(const kernel_impl_params& params);
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        const auto desc = node.as<scaled_dot_product_attention>().get_primitive();
        static constexpr std::array supported_q_types = {
            ov::element::f32,
            ov::element::f16,
        };
        static constexpr std::array supported_kv_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i8,
        };
        const auto& q_layout = node.get_input_layout(0);
        const auto& k_layout = node.get_input_layout(1);
        const auto& v_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!everyone_is(format::bfyx, q_layout.format, k_layout.format, v_layout.format, out_layout.format)) {
            return false;
        }

        if (!one_of(k_layout.data_type, supported_kv_types) || !one_of(v_layout.data_type, supported_kv_types)) {
            return false;
        }

        if (!one_of(q_layout.data_type, supported_q_types) || !one_of(out_layout.data_type, supported_q_types)) {
            return false;
        }

        auto dim_size = desc->input_k_transpose_order.size();
        auto k_head_size = k_layout.get_partial_shape()[desc->input_k_transpose_order[dim_size - 1]];
        if (k_head_size.is_dynamic()) {
            return false;
        }

        const bool use_asymmetric_quantization =
            desc->quantization_attributes.quantization_type == ov::op::internal::DynamicQuantize::QuantizationType::Asymmetric;
        const bool combine_scales_and_zp = desc->quantization_attributes.output_storage_type != ov::op::internal::DynamicQuantize::OutputStorageType::Planar;

        auto p = node.get_kernel_impl_params();
        return !use_asymmetric_quantization || combine_scales_and_zp || supports_micro_sdpa(*p);
    }
};

}  // namespace ov::intel_gpu::ocl
