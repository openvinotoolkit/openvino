// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "fully_connected_inst.h"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "registry/implementation_manager.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

#define CM_FC_LOG_AND_RETURN_FALSE(node, reason) do {                                                        \
    GPU_DEBUG_TRACE << (node).id() << " : Do not select cm::fully_connected::woq_u2 (" << reason << ")" << std::endl; \
    return false;                                                                                          \
} while (0)

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

        if (!check_cm_jit_support(engine, config))
            CM_FC_LOG_AND_RETURN_FALSE(node, "CM jit not supported on this device");
        if (info.arch != gpu_arch::xe2 && info.arch != gpu_arch::xe3)
            CM_FC_LOG_AND_RETURN_FALSE(node, "unsupported arch (requires Xe2/Xe3)");
        // Deliberately not gated on config.get_use_cm(): that flag is a blanket switch for every CM
        // kernel (PA/SDPA/LSTM/FC); this impl dispatches on its own whenever all other checks below
        // pass, without requiring OV_GPU_USE_CM to be set.
        if (info.max_local_mem_size < required_slm_bytes)
            CM_FC_LOG_AND_RETURN_FALSE(node, "insufficient SLM: " << info.max_local_mem_size << " < " << required_slm_bytes);
        if (node.has_fused_primitives())
            CM_FC_LOG_AND_RETURN_FALSE(node, "fused primitives not supported");

        const auto& fc_node = node.as<fully_connected>();
        const auto& prim = fc_node.get_primitive();
        if (!prim->compressed_weights)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights not compressed");
        if (!prim->decompression_scale.is_valid())
            CM_FC_LOG_AND_RETURN_FALSE(node, "no decompression scale");
        if (prim->bias.is_valid())
            CM_FC_LOG_AND_RETURN_FALSE(node, "bias not supported");
        if (prim->dynamic_quantized_activation)
            CM_FC_LOG_AND_RETURN_FALSE(node, "dynamic quantized activation not supported");
        if (!prim->weights_transposed)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights not transposed");
        if (prim->weights_rank != 2)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights_rank != 2: " << prim->weights_rank);

        const auto in_layouts = node.get_input_layouts();
        const auto& in = in_layouts[0];
        const auto& wei = in_layouts[1];
        const auto& scale = in_layouts[2];
        const auto out = node.get_output_layout(0);

        if (in.data_type != data_types::f16)
            CM_FC_LOG_AND_RETURN_FALSE(node, "activation dtype != f16: " << in.data_type);
        if (in.format != format::bfyx || in.data_padding)
            CM_FC_LOG_AND_RETURN_FALSE(node, "activation not plain bfyx");
        if (wei.data_type != data_types::u2)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights dtype != u2: " << wei.data_type);
        if (wei.format != format::bfyx || wei.data_padding || !wei.is_static())
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights not static plain bfyx");
        if (out.data_type != data_types::f16 && out.data_type != data_types::f32)
            CM_FC_LOG_AND_RETURN_FALSE(node, "output dtype not f16/f32: " << out.data_type);
        if (out.format != format::bfyx || out.data_padding)
            CM_FC_LOG_AND_RETURN_FALSE(node, "output not plain bfyx");

        // Weights [N, K] (static even when M is dynamic).
        const auto wshape = wei.get_shape();
        if (wshape.size() != 2)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weight shape rank != 2: " << wshape.size());
        const auto N = static_cast<int64_t>(wshape[0]);
        const auto K = static_cast<int64_t>(wshape[1]);
        if (K % group_size != 0)
            CM_FC_LOG_AND_RETURN_FALSE(node, "K % 64 != 0: K=" << K);
        if (N % 32 != 0)
            CM_FC_LOG_AND_RETURN_FALSE(node, "N % 32 != 0: N=" << N);
        const auto KG = K / group_size;

        // Activations: last dim is K.
        const auto& in_pshape = in.get_partial_shape();
        if (in_pshape.rank().is_dynamic() || in_pshape[in_pshape.size() - 1].is_dynamic() ||
            in_pshape[in_pshape.size() - 1].get_length() != K)
            CM_FC_LOG_AND_RETURN_FALSE(node, "activation last dim != K or dynamic");

        auto is_per_group = [&](const cldnn::layout& l) {
            if (!l.is_static() || l.format != format::bfyx || l.data_padding)
                return false;
            const auto s = l.get_shape();
            size_t total = 1;
            for (auto d : s)
                total *= d;
            return s.size() >= 2 && static_cast<int64_t>(s[0]) == N && static_cast<int64_t>(total) == N * KG;
        };

        if (scale.data_type != data_types::f16)
            CM_FC_LOG_AND_RETURN_FALSE(node, "scale dtype != f16: " << scale.data_type);
        if (!is_per_group(scale)) {
            size_t total = 1;
            for (auto d : scale.get_shape())
                total *= d;
            CM_FC_LOG_AND_RETURN_FALSE(node, "scale shape/format mismatch: format=" << scale.format
                                        << " total_elems=" << total << " expected(N*K/64)=" << (N * KG)
                                        << " N=" << N << " K=" << K
                                        << " implied_group_size=" << (total > 0 ? (N * K) / static_cast<int64_t>(total) : -1));
        }

        // Zero points: a per-group u8 tensor only.
        if (!prim->decompression_zero_point.is_valid() || in_layouts.size() < 4)
            CM_FC_LOG_AND_RETURN_FALSE(node, "no decompression zero point");
        const auto& zp = in_layouts[3];
        if (zp.data_type != data_types::u8)
            CM_FC_LOG_AND_RETURN_FALSE(node, "zero point dtype != u8: " << zp.data_type);
        if (!is_per_group(zp)) {
            size_t total = 1;
            for (auto d : zp.get_shape())
                total *= d;
            CM_FC_LOG_AND_RETURN_FALSE(node, "zero point shape/format mismatch: format=" << zp.format
                                        << " total_elems=" << total << " expected(N*K/64)=" << (N * KG)
                                        << " N=" << N << " K=" << K
                                        << " implied_group_size=" << (total > 0 ? (N * K) / static_cast<int64_t>(total) : -1));
        }

        return true;
    }
};

}  // namespace ov::intel_gpu::cm
