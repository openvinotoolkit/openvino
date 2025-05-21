// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct GroupNormalizationFsv16Opt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::fsv16_opt")
    explicit GroupNormalizationFsv16Opt(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
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
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        if (in0_layout.is_static() && out_layout.is_static()) {
            // no support for spatial paddings in static case
            if (in0_layout.data_padding._lower_size[3] > 0 || in0_layout.data_padding._lower_size[2] > 0 || in0_layout.data_padding._upper_size[3] > 0 ||
                in0_layout.data_padding._upper_size[2] > 0) {
                return false;
            }

            if (!fused_ops_are_one_of<eltwise, activation>(node.get_fused_primitives())) {
                return false;
            }
        }

        constexpr size_t fsv = 16;
        // feature paddings should be multiples of fsv.
        return in0_layout.data_padding._lower_size[1] % fsv == 0;
    }
};

}  // namespace ov::intel_gpu::ocl
