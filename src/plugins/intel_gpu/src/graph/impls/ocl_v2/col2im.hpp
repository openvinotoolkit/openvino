// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct Col2Im : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::col2im")
    explicit Col2Im(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {data_types::f16, data_types::f32, data_types::u8, data_types::i8};

        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx};

        if (node.has_fused_primitives()) {
            return false;
        }

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
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
