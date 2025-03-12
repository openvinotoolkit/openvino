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

struct CTCLoss : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::ctc_loss")
    explicit CTCLoss(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_types = {data_types::f16, data_types::f32};

        static constexpr std::array supported_fmts = {format::bfyx,
                                                      format::b_fs_yx_fsv16,
                                                      format::b_fs_yx_fsv32,
                                                      format::bs_fs_yx_bsv16_fsv16,
                                                      format::bs_fs_yx_bsv32_fsv32,
                                                      format::bs_fs_yx_bsv32_fsv16};

        if (node.has_fused_primitives()) {
            return false;
        }

        for (const auto& in_l : node.get_input_layouts()) {
            if (!one_of(in_l.format, supported_fmts)) {
                return false;
            }
        }

        if (!one_of(node.get_input_layout(0).data_type, supported_types)) {
            return false;
        }

        for (const auto& out_l : node.get_output_layouts()) {
            if (!one_of(out_l.format, supported_fmts) || !one_of(out_l.data_type, supported_types)) {
                return false;
            }
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
