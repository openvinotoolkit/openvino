// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "impls/registry/implementation_manager.hpp"
#include "program_node.h"

#include <memory>

namespace ov::intel_gpu::ocl {

struct GroupNormalizationFsv16Opt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::group_norm::fsv16_opt")
    GroupNormalizationFsv16Opt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    bool validate_impl(const program_node& node) const override {
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
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts))
            return false;

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types))
            return false;

        if (in0_layout.is_dynamic() || out_layout.is_dynamic())
            return true;

        // no support for spatial paddings
        if (in0_layout.data_padding._lower_size[3] > 0 || in0_layout.data_padding._lower_size[2] > 0 ||
            in0_layout.data_padding._upper_size[3] > 0 || in0_layout.data_padding._upper_size[2] > 0) {
            return false;
        }

        static constexpr size_t fsv = 16;

        // feature paddings should be multiples of fsv.
        if (in0_layout.data_padding._lower_size[1] % fsv != 0) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
