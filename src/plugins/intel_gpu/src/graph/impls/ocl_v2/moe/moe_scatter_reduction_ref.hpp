// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "moe_scatter_reduction_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {
struct MoeScatterReductionRef : public MoeScatterReductionBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe_scatter_reduction::ref")
    explicit MoeScatterReductionRef(shape_types shape_type, ValidateFunc vf = nullptr) : MoeScatterReductionBase(shape_type, std::move(vf)) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::i32,
            ov::element::i64,
            ov::element::i8,
            ov::element::u8,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        const auto& input_pshapes = in0_layout.get_partial_shape();

        if (input_pshapes.rank().get_length() != 3 || input_pshapes[2].is_dynamic()) {
            return false;
        }

        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        return true;
    }
};
}  // namespace ov::intel_gpu::ocl
