// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/quantize.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

struct GatherRef : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::gather_ref")
    explicit GatherRef(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_data_types =
            {ov::element::f16, ov::element::f32, ov::element::i32, ov::element::u8, ov::element::i8, ov::element::u4, ov::element::i4};
        static constexpr std::array supported_indices_types = {ov::element::i32};

        static constexpr std::array supported_static_fmts = {
            format::fyxb,
            format::yxfb,
            format::byxf,
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
            format::b_fs_yx_fsv4,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::b_fs_zyx_fsv16,
            format::b_fs_zyx_fsv32,
            format::bs_fs_yx_bsv4_fsv2,
            format::bs_fs_yx_bsv4_fsv4,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv32,
            format::fs_b_yx_fsv32,
        };
        static constexpr std::array supported_dyn_fmts = {
            format::bfyx,
            format::bfzyx,
            format::bfwzyx,
        };

        if (!fused_ops_are_one_of<eltwise, quantize, activation, reorder>(node.get_fused_primitives())) {
            return false;
        }

        const auto& data_l = node.get_input_layout(0);
        const auto& ind_l = node.get_input_layout(1);
        const auto& out_l = node.get_output_layout(0);

        if (out_l.is_static()) {
            if (!one_of(data_l.format, supported_static_fmts) || !one_of(ind_l.format, supported_static_fmts) || !one_of(out_l.format, supported_static_fmts)) {
                return false;
            }
        } else {
            if (!one_of(data_l.format, supported_dyn_fmts) || !one_of(ind_l.format, supported_dyn_fmts) || !one_of(out_l.format, supported_dyn_fmts)) {
                return false;
            }
        }

        if (!one_of(data_l.data_type, supported_data_types) || !one_of(ind_l.data_type, supported_indices_types) ||
            !one_of(out_l.data_type, supported_data_types)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
