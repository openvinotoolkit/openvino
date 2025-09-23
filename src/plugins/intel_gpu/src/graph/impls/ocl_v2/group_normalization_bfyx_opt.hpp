// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "group_normalization_base.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationBfyxOpt : public GroupNormalizationBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::bfyx_opt")
    explicit GroupNormalizationBfyxOpt(shape_types shape_type, ValidateFunc vf = nullptr) : GroupNormalizationBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_input_fmts = {
            format::bfyx,
            format::bfzyx,
        };

        static constexpr std::array supported_output_fmts = {
            format::bfyx,
            format::bfzyx,
            format::b_fs_yx_fsv16,
        };

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::u8,
            ov::element::i8,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& in1_layout = node.get_input_layout(1);
        const auto& in2_layout = node.get_input_layout(2);
        const auto& out_layout = node.get_output_layout(0);
        if (!(one_of(in0_layout.format, supported_input_fmts) ||
              one_of(in1_layout.format, supported_input_fmts) ||
              one_of(in2_layout.format, supported_input_fmts)) ||
            !one_of(out_layout.format, supported_output_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        if (!fused_ops_are_one_of<eltwise, activation, reorder>(node.get_fused_primitives())) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
