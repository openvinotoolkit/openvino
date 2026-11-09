// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct Histc : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::histc")
    explicit Histc(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {data_types::f32, data_types::f16, data_types::i32, data_types::i8, data_types::u8};

        if (node.has_fused_primitives()) {
            return false;
        }

        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        // Output is the input type or i32 (preserve_input_type = false).
        if (!one_of(in_layout.data_type, supported_types) || !one_of(out_layout.data_type, {in_layout.data_type, data_types::i32})) {
            return false;
        }

        // Elements are read linearly, so the layout must not contain padding or blocked dimensions.
        return format::is_simple_data_format(in_layout.format) && !in_layout.data_padding && format::is_simple_data_format(out_layout.format) &&
               !out_layout.data_padding;
    }
};

}  // namespace ov::intel_gpu::ocl
