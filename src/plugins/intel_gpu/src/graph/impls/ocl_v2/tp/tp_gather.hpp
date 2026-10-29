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

struct TPGather : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::tp_gather")
    explicit TPGather(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node,
                                                              const RuntimeParams& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {data_types::f16, data_types::bf16, data_types::f32};
        static constexpr std::array supported_fmts = {format::bfyx, format::bfzyx, format::bfwzyx};

        if (node.has_fused_primitives()) {
            return false;
        }

        const auto& in_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }
        return one_of(in_layout.data_type, supported_types) && in_layout.data_type == out_layout.data_type;
    }
};

}  // namespace ov::intel_gpu::ocl
