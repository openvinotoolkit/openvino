// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "intel_gpu/primitives/activation.hpp"
#include "intel_gpu/primitives/eltwise.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"
#include "intel_gpu/runtime/device_info.hpp"
#include "openvino/core/type/element_type.hpp"
#include "program_node.h"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

// CM-kernel-based GGUF FC implementation (prefill GEMM on C-for-Metal dpas kernels).
//
// Handles the six shuffle-eligible GGUF formats (Q4_0 / Q4_1 / Q4_K / Q5_K / Q6_K / Q8_0).
// Requires the weight to be in the SG-transposed plane-separated layout produced by
// RepackGGUFWeightsShuffle (same prerequisite as the OCL shuffle-GEMV path).
//
// Phase routing at execute time:
//   Prefill (M > threshold): C-for-Metal systolic dpas GEMM (gemm_q*_full.cm).
//     The CM kernels accumulate in fp32 and store fp16 or fp32 output selected
//     by JIT (enforced in support_shapes).  Activation input is always fp16 on
//     the wire: an fp32 runtime activation is first converted into a scratch
//     fp16 buffer by a tiny conversion kernel (convert_act_f32_to_f16.cm), see
//     FCGGUFCmImpl::execute_cm_gemm in fc_gguf_cm.cpp.
//   Decode  (M <= threshold): OCL sub-group-block-read shuffle GEMV (fc_gguf_q*k_sg.cl).
//     This kernel JIT-selects its activation type directly and natively
//     supports fp16/fp32 activation without any conversion pre-pass.
struct FCGGUFCm : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::fc_gguf_cm")

    static constexpr std::array<element::Type_t, 6> kSupportedFormats = {
        element::Type_t::gguf_q4_0,
        element::Type_t::gguf_q4_1,
        element::Type_t::gguf_q4_k,
        element::Type_t::gguf_q5_k,
        element::Type_t::gguf_q6_k,
        element::Type_t::gguf_q8_0,
    };

    static bool is_supported_format(element::Type_t t) {
        for (auto s : kSupportedFormats) {
            if (s == t)
                return true;
        }
        return false;
    }

    explicit FCGGUFCm(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const RuntimeParams& params) const override;

    // Accept fully_connected nodes with shuffle-eligible GGUF weight.
    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        if (!check_cm_jit_support(engine, config) || !engine.get_device_info().supports_immad || !config.get_use_cm()) {
            return false;
        }
        const auto& desc = *node.get_kernel_impl_params()->typed_desc<fully_connected>();
        if (!desc.compressed_weights) {
            return false;
        }
        if (const char* env = std::getenv("OV_GPU_GGUF_GEMM_CM")) {
            if (std::atol(env) == 0)
                return false;
        }
        const auto& in0 = node.get_input_layout(0);  // activation
        const auto& in1 = node.get_input_layout(1);  // weight
        if (in0.data_type != data_types::f16 && in0.data_type != data_types::f32) {
            return false;
        }
        if (!element::is_gguf_block(in1.data_type)) {
            return false;
        }
        // CM GEMM supports fp16 and fp32 output through the CM_OUTPUT_F32 JIT
        // switch.  Keep the output type explicit, since other element types
        // are not covered by the output-store code in the CM kernels.
        const auto output_type = node.get_output_layout(0).data_type;
        if (output_type != data_types::f16 && output_type != data_types::f32) {
            return false;
        }
        return is_supported_format(in1.data_type) && support_shapes(*node.get_kernel_impl_params());
    }

    // Accept when the weight is static with shuffle-applicable geometry.
    [[nodiscard]] bool support_shapes(const kernel_impl_params& params) const override {
        const auto& in1 = params.get_input_layout(1);
        if (in1.is_dynamic() || !element::is_gguf_block(in1.data_type)) {
            return false;
        }
        if (!is_supported_format(in1.data_type)) {
            return false;
        }
        // If the shuffle transform is disabled, weights are in native (non-shuffled)
        // GGUF block format — the CM GEMM kernels require the SG-transposed layout.
        if (const char* env = std::getenv("OV_GPU_GGUF_SHUFFLE")) {
            if (std::atol(env) == 0)
                return false;
        }
        // Match complete CM channel tiles: ROW_GROUPS=4 on Xe3, 2 otherwise.
        const size_t N = in1.get_shape()[0];
        const size_t K = in1.get_shape()[1];
        const auto channel_alignment = params.get_device_info().arch >= gpu_arch::xe3 ? size_t{64} : size_t{32};
        if (N % channel_alignment != 0 || K % 256 != 0) {
            return false;
        }
        // CM GEMM kernels select fp16/fp32 output stores through JIT.
        const auto& out = params.get_output_layout(0);
        if (out.data_type != data_types::f16 && out.data_type != data_types::f32) {
            return false;
        }
        // Verify all non-reorder fused ops are CM-supported types.
        // Unknown op types (e.g. unsupported activations) must fall through to OCL.
        if (params.has_fused_primitives()) {
            for (const auto& fd : params.fused_desc) {
                if (fd.is_type<cldnn::reorder>())
                    continue;  // layout-only, always OK
                if (fd.is_type<cldnn::activation>()) {
                    const auto act = fd.typed_desc<cldnn::activation>();
                    if (fd.has_outer_dep() || (act->activation_function != cldnn::activation_func::relu &&
                                               act->activation_function != cldnn::activation_func::swish)) {
                        return false;
                    }
                    continue;
                }
                if (fd.is_type<cldnn::eltwise>()) {
                    const auto elt = fd.typed_desc<cldnn::eltwise>();
                    if (!fd.has_outer_dep() || (elt->mode != cldnn::eltwise_mode::sum &&
                                                elt->mode != cldnn::eltwise_mode::prod) ||
                        elt->input.size() != 2 || !elt->coefficients.empty() || !elt->stride.empty()) {
                        return false;
                    }
                    const auto& operand = params.get_input_layout(fd.outer_dep_start_idx);
                    if (operand.is_dynamic() || out.is_dynamic() || operand.get_shape() != out.get_shape() ||
                        operand.format != format::bfyx || out.format != format::bfyx ||
                        operand.data_padding || out.data_padding ||
                        (operand.data_type != data_types::f16 && operand.data_type != data_types::f32)) {
                        return false;
                    }
                    continue;
                }
                return false;  // unsupported fused op type → fall back to OCL
            }
        }
        // Dynamic activation shape is fine.  fp32 activation storage is also
        // fine: FCGGUFCmImpl converts it to fp16 with a small pre-pass kernel
        // before dispatching the big GEMM kernel (see fc_gguf_cm.cpp).
        return true;
    }
};

}  // namespace ov::intel_gpu::cm
