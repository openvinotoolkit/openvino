// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "moe_3gemm_base.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

struct moe_3gemm_swiglu_opt : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ocl::moe::moe_3gemm_swiglu_opt")
    explicit moe_3gemm_swiglu_opt(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, std::move(vf)) {}
    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        static constexpr std::array supported_fmts = {
            format::bfyx,
        };

        // TODO(MOE): support more precision
        static constexpr std::array supported_types = {
            ov::element::f16,
        };

        const auto& in0_layout = node.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES));
        const auto& out_layout = node.get_output_layout(0);
        if (!one_of(in0_layout.format, supported_fmts) || !one_of(out_layout.format, supported_fmts)) {
            return false;
        }

        if (!one_of(in0_layout.data_type, supported_types) || !one_of(out_layout.data_type, supported_types)) {
            return false;
        }

        // Only support weight: u4, i8, u8
        static constexpr std::array supported_wei_type = {
            ov::element::u4,
            ov::element::u8,
            ov::element::i8,
        };
        const auto& wei_layout = node.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0));
        if (!one_of(wei_layout.data_type, supported_wei_type)) {
            return false;
        }

        // Only support scale: f16
        static constexpr std::array supported_scale_type = {
            ov::element::f16,
        };
        const auto& scale_layout = node.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SCALE_0));
        if (!one_of(scale_layout.data_type, supported_scale_type)) {
            return false;
        }

        // Only support zp: u4, u8
        static constexpr std::array supported_zp_type = {
            ov::element::u4,
            ov::element::u8,
        };
        const auto& zp_layout = node.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::ZP_0));
        if (!one_of(zp_layout.data_type, supported_zp_type)) {
            return false;
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
