// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct BevPoolV2 : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::bevpool_v2")
    explicit BevPoolV2(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_float_types = {data_types::f16, data_types::f32};
        static constexpr std::array supported_index_types = {data_types::i32, data_types::i64, data_types::u32};
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        if (node.has_fused_primitives()) {
            return false;
        }

        if (node.get_input_layouts().size() != 4 || node.get_output_layouts().size() != 1) {
            return false;
        }

        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& in2_layout = node.get_input_layout(2);
        const auto& in3_layout = node.get_input_layout(3);
        const auto& out_layout = node.get_output_layout(0);

        if (!one_of(in0_layout.data_type, supported_float_types) ||
            !one_of(in1_layout.data_type, supported_float_types) ||
            !one_of(in2_layout.data_type, supported_index_types) ||
            !one_of(in3_layout.data_type, supported_index_types) ||
            !one_of(out_layout.data_type, supported_float_types)) {
            return false;
        }

        if (!one_of(in0_layout.format, supported_fmts) ||
            !one_of(in1_layout.format, supported_fmts) ||
            !one_of(in2_layout.format, supported_fmts) ||
            !one_of(in3_layout.format, supported_fmts) ||
            !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        for (auto& input : node.get_input_layouts()) {
            if (input.data_padding)
                return false;
        }

        for (auto& output : node.get_output_layouts()) {
            if (output.data_padding)
                return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
