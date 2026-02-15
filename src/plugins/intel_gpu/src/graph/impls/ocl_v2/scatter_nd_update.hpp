// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct ScatterNDUpdate : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::scatter_nd_update")
    explicit ScatterNDUpdate(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_inout_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
            ov::element::i8,
            ov::element::u8,
        };

        static constexpr std::array supported_indices_types = {
            ov::element::i32,
            ov::element::i64,
        };

        const auto& in0_layout = node.get_input_layout(0);  // data
        const auto& in1_layout = node.get_input_layout(1);  // indices
        const auto& in2_layout = node.get_input_layout(2);  // updates
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.data_type, supported_inout_types) || !one_of(in1_layout.data_type, supported_indices_types) ||
            !one_of(in2_layout.data_type, supported_inout_types) || !one_of(out_layout.data_type, supported_inout_types)) {
            return false;
        }

        if (!fused_ops_are_one_of<eltwise, activation, quantize>(node.get_fused_primitives())) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
