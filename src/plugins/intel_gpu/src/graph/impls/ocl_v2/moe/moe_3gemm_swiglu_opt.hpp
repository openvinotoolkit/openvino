// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
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

        using Idx = MOE3GemmInputIndex;
        const auto dt_of = [&](Idx idx) {
            return ov::element::Type(node.get_input_layout(static_cast<size_t>(idx)).data_type);
        };
        const auto is_signed = [](ov::element::Type dt) {
            return dt == ov::element::i4 || dt == ov::element::i8;
        };

        // Only support weight: u4, i4, u8, i8
        static constexpr std::array supported_wei_type = {
            ov::element::u4,
            ov::element::i4,
            ov::element::u8,
            ov::element::i8,
        };
        // Only support scale: f16
        static constexpr std::array supported_scale_type = {
            ov::element::f16,
        };
        // Only support zp: u4, i4, u8, i8 (skip check for symmetric quantization where ZP is element::dynamic placeholder)
        static constexpr std::array supported_zp_type = {
            ov::element::u4,  // asym-quant type
            ov::element::i4,  // sym-quant type
            ov::element::u8,  // asym-quant type
            ov::element::i8,  // sym-quant type
        };
        const auto& config = std::static_pointer_cast<const moe_3gemm_fused_compressed>(node.get_primitive())->_config;

        static constexpr std::array weights = {Idx::WEIGHT_0, Idx::WEIGHT_1, Idx::WEIGHT_2};
        static constexpr std::array scales = {Idx::SCALE_0, Idx::SCALE_1, Idx::SCALE_2};
        static constexpr std::array zps = {Idx::ZP_0, Idx::ZP_1, Idx::ZP_2};
        for (size_t i = 0; i < weights.size(); ++i) {
            const auto wei_dt = dt_of(weights[i]);
            if (!one_of(wei_dt, supported_wei_type) || !one_of(dt_of(scales[i]), supported_scale_type)) {
                return false;
            }
            // WEIGHT_IS_SIGNED is derived from WEIGHT_0 and applied to all three GEMMs.
            if (is_signed(wei_dt) != is_signed(dt_of(weights[0]))) {
                return false;
            }
            if (config.has_zp) {
                const auto zp_dt = dt_of(zps[i]);
                // The kernel strides zp by the weight's bit width (MOE_EXPERT_ZP_BYTES), so they must match.
                if (!one_of(zp_dt, supported_zp_type) || zp_dt.bitwidth() != wei_dt.bitwidth()) {
                    return false;
                }
            }
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::ocl
