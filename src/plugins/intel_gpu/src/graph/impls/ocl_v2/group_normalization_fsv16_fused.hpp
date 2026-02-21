// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "group_normalization_base.hpp"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct GroupNormalizationFsv16Fused : public GroupNormalizationBase {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::fsv16_fused")
    explicit GroupNormalizationFsv16Fused(shape_types shape_type, ValidateFunc vf = nullptr) : GroupNormalizationBase(shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_input_fmts = {
            format::b_fs_yx_fsv16,
        };

        static constexpr std::array supported_output_fmts = {
            format::b_fs_yx_fsv16,
        };

        static constexpr std::array supported_types = {
            ov::element::f32,
            ov::element::f16,
            ov::element::u8,
            ov::element::i8,
        };

        const auto& in0_layout = node.get_input_layout(0);
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_input_fmts) || !one_of(out_layout.format, supported_output_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        if (in0_layout.is_static() && out_layout.is_static()) {
            if (!fused_ops_are_one_of<eltwise, activation, reorder>(node.get_fused_primitives())) {
                return false;
            }
        }

        // padding is not supported
        if (in0_layout.data_padding != padding() || out_layout.data_padding != padding()) {
            return false;
        }

        constexpr int64_t fsv = 16;
        // feature count needs to be static for following checks
        if (in0_layout.get_partial_shape()[1].is_dynamic())
            return false;
        // feature count should be a multiple of fsv
        if (in0_layout.feature() % fsv != 0)
            return false;
        // group size must be a divisor of fsv
        const auto group_size = in0_layout.feature() / std::static_pointer_cast<const group_normalization>(node.get_primitive())->num_groups;
        return group_size <= fsv && fsv % group_size == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
