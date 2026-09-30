// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::cm {

// Weight layout woq_u2_gemm_dual.cm is built for (its WLAYOUT). A compressed FC always provides n_major:
// weights [N, K/4], scales / zero points [N, K/64]. group_major -- weights [K/64, N, 16 B], scales / zero
// points [K/64, N], the kernel's other path -- is selectable for tests only: the buffers are passed
// through unchanged, so whoever selects it must fill them in group-major byte order. Read when the
// kernel is generated (network build).
enum class WoqU2WeightLayout : int { group_major = 0, n_major = 1 };
inline WoqU2WeightLayout& woq_u2_weight_layout_for_tests() {
    static WoqU2WeightLayout layout = WoqU2WeightLayout::n_major;
    return layout;
}

// u2 weight-only-quantized fully connected layer on XMX/DPAS (woq_u2_gemm_dual.cm, built with WLAYOUT=1:
// N-major weights and scales / zero points, no epilogue).
//
// Selected only for: Xe2 or Xe3, CM enabled, >= 96 KB SLM; f16 activations without padding; u2 weights
// [N, K] (weights_transposed); f16 decompression scales [N, K/64] bfyx (group size 64); u8 decompression
// zero points [N, K/64] bfyx (required: the kernel has no scalar / f16 / absent zero-point path); no
// bias, no fused ops, no dynamically quantized activations; f16 or f32 output (OUT_F16);
// K % 64 == 0 and N % 32 == 0. M (the product of the leading dims) may be dynamic. Everything else falls
// back to the other fully_connected implementations (for u2: the OCL reference kernel).
// prepare_quantization keeps per-group scales / zero points of u2 FCs in bfyx for this kernel.
struct FullyConnectedWoqU2ImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("cm::fully_connected::woq_u2")
    explicit FullyConnectedWoqU2ImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr)
        : ImplementationManager(impl_types::cm, shape_type, std::move(vf)) {}

    static constexpr uint64_t required_slm_bytes = 3 * (2 * 256 * 32 + 64 * 256);  // matches SLM_BYTES in woq_u2_gemm_dual.cm
    static constexpr int64_t group_size = 64;

    [[nodiscard]] in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        // Everything plain bfyx: the kernel's N-major layout needs scales / zero points physically [N, K/64].
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::bfyx);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::bfyx);
        return {in_fmts, out_fmts};
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    [[nodiscard]] bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());

        auto& engine = node.get_program().get_engine();
        const auto& config = node.get_program().get_config();
        const auto& info = engine.get_device_info();

        if (!check_cm_jit_support(engine, config) || (info.arch != gpu_arch::xe2 && info.arch != gpu_arch::xe3) || !config.get_use_cm())
            return false;
        if (info.max_local_mem_size < required_slm_bytes)
            return false;
        if (node.has_fused_primitives())
            return false;

        const auto& fc_node = node.as<fully_connected>();
        const auto& prim = fc_node.get_primitive();
        if (!prim->compressed_weights || !prim->decompression_scale.is_valid() || prim->bias.is_valid() ||
            prim->dynamic_quantized_activation || !prim->weights_transposed || prim->weights_rank != 2)
            return false;

        const auto in_layouts = node.get_input_layouts();
        const auto& in = in_layouts[0];
        const auto& wei = in_layouts[1];
        const auto& scale = in_layouts[2];
        const auto out = node.get_output_layout(0);

        if (in.data_type != data_types::f16 || in.format != format::bfyx || in.data_padding)
            return false;
        if (wei.data_type != data_types::u2 || wei.format != format::bfyx || wei.data_padding || !wei.is_static())
            return false;
        if (out.data_type != data_types::f16 && out.data_type != data_types::f32)
            return false;
        if (out.format != format::bfyx || out.data_padding)
            return false;

        // Weights [N, K] (static even when M is dynamic).
        const auto wshape = wei.get_shape();
        if (wshape.size() != 2)
            return false;
        const auto N = static_cast<int64_t>(wshape[0]);
        const auto K = static_cast<int64_t>(wshape[1]);
        if (K % group_size != 0 || N % 32 != 0)
            return false;
        const auto KG = K / group_size;

        // Activations: last dim is K.
        const auto& in_pshape = in.get_partial_shape();
        if (in_pshape.rank().is_dynamic() || in_pshape[in_pshape.size() - 1].is_dynamic() ||
            in_pshape[in_pshape.size() - 1].get_length() != K)
            return false;

        auto is_per_group = [&](const cldnn::layout& l) {
            if (!l.is_static() || l.format != format::bfyx || l.data_padding)
                return false;
            const auto s = l.get_shape();
            size_t total = 1;
            for (auto d : s)
                total *= d;
            return s.size() >= 2 && static_cast<int64_t>(s[0]) == N && static_cast<int64_t>(total) == N * KG;
        };

        if (scale.data_type != data_types::f16 || !is_per_group(scale))
            return false;

        // Zero points: a per-group u8 tensor only.
        if (!prim->decompression_zero_point.is_valid() || in_layouts.size() < 4)
            return false;
        const auto& zp = in_layouts[3];
        if (zp.data_type != data_types::u8 || !is_per_group(zp))
            return false;

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
