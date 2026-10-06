// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "activation_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
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

// Dependencies of the FC: 0 = input, 1 = weights, [2 = bias], then decompression scale and zero point,
// then the outer dependencies of fused primitives.
constexpr size_t woq_u2_bias_idx = 2;
inline size_t woq_u2_scale_idx(bool has_bias) { return has_bias ? 3 : 2; }
inline size_t woq_u2_zp_idx(bool has_bias) { return woq_u2_scale_idx(has_bias) + 1; }

// The epilogue the CM u2 kernels apply (their epi_mode) and the FC dependency that provides its tensor
// (`epi`), derived from the FC's bias and fused primitives. Supported, and nothing else:
//   no bias, nothing fused                                    -> mode 0: C = acc
//   FC bias                                                   -> mode 1: C = acc + bias
//   fused eltwise sum with one outer operand                  -> mode 1: C = acc + operand
//   fused swish (beta 1) then eltwise prod with an operand    -> mode 2: C = swish(acc) * operand (SwiGLU)
// The operand itself is checked by woq_u2_check_epi_operand (dtype; shape like the output, or one row [N]
// when the output is one row too).
struct WoqU2Epilogue {
    int32_t mode = 0;      // kernel epi_mode
    int32_t dep = -1;      // FC dependency index of the epilogue tensor, -1 if none
    std::string error;     // non-empty: unsupported combination (the reason)
    bool ok() const { return error.empty(); }
};

inline WoqU2Epilogue woq_u2_epilogue(const std::vector<fused_primitive_desc>& fused, bool has_bias) {
    WoqU2Epilogue e;
    if (fused.empty()) {
        if (has_bias) {
            e.mode = 1;
            e.dep = static_cast<int32_t>(woq_u2_bias_idx);
        }
        return e;
    }
    if (has_bias) {
        e.error = "bias together with fused primitives";
        return e;
    }
    // A fused eltwise of `mode` taking the FC's (fused) result and exactly one outer operand, no strides or
    // coefficients.
    auto is_binary = [](const fused_primitive_desc& fd, eltwise_mode mode) {
        if (!fd.is_type<eltwise>() || !fd.has_outer_dep() || fd.total_num_deps != 2)
            return false;
        const auto prim = fd.typed_desc<eltwise>();
        if (prim->mode != mode || !prim->stride.empty())
            return false;
        for (auto c : prim->coefficients)
            if (c != 1.0f)
                return false;
        return true;
    };
    if (fused.size() == 1 && is_binary(fused[0], eltwise_mode::sum)) {
        e.mode = 1;
        e.dep = fused[0].outer_dep_start_idx;
        return e;
    }
    if (fused.size() == 2 && fused[0].is_type<activation>() && !fused[0].has_outer_dep() && is_binary(fused[1], eltwise_mode::prod)) {
        const auto act = fused[0].typed_desc<activation>();
        // The kernel's swish is x / (1 + exp(-x)): beta must be 1 and constant.
        if (act->activation_function == activation_func::swish && !act->additional_params_input.is_valid() && act->additional_params.a == 1.0f) {
            e.mode = 2;
            e.dep = fused[1].outer_dep_start_idx;
            return e;
        }
    }
    std::ostringstream os;
    os << "fused primitives not supported:";
    for (const auto& fd : fused)
        os << " " << fd.desc->id;
    os << " (supported: one eltwise sum, or swish(beta 1) + eltwise prod)";
    e.error = os.str();
    return e;
}

// True if all dims but the last are static 1: a single row [.., 1, N].
inline bool woq_u2_is_one_row(const cldnn::layout& l) {
    const auto& ps = l.get_partial_shape();
    for (size_t i = 0; i + 1 < ps.size(); i++)
        if (!ps[i].is_static() || ps[i].get_length() != 1)
            return false;
    return ps.size() >= 1;
}

// Empty if `epi` can be the kernels' epilogue tensor for output `out` with N columns, else the reason.
inline std::string woq_u2_check_epi_operand(const cldnn::layout& epi, const cldnn::layout& out, int64_t N) {
    if (epi.data_type != data_types::f16 && epi.data_type != data_types::f32)
        return "epilogue tensor dtype not f16/f32";
    if (!format::is_simple_data_format(epi.format) || epi.data_padding)
        return "epilogue tensor not plain / padded";
    const auto& ps = epi.get_partial_shape();
    if (ps.rank().is_dynamic() || ps.size() == 0 || !ps[ps.size() - 1].is_static() || ps[ps.size() - 1].get_length() != N)
        return "epilogue tensor last dim != N";
    // The kernels read epi[m * N + n]: a one-row operand (bias, per-column post-op) is only that when the
    // output is one row as well (M == 1); for more rows it would need broadcasting, which they do not do.
    if (woq_u2_is_one_row(epi)) {
        if (woq_u2_is_one_row(out))
            return {};
        return "one-row (per-column) epilogue tensor with more than one output row (M must be 1)";
    }
    // Full tensor: same shape as the output, dimension by dimension (dynamic only where the output is).
    const auto& os = out.get_partial_shape();
    if (os.rank().is_dynamic() || os.size() != ps.size())
        return "epilogue tensor rank != output rank";
    for (size_t i = 0; i < ps.size(); i++) {
        if (ps[i].is_static() != os[i].is_static() || (ps[i].is_static() && ps[i].get_length() != os[i].get_length()))
            return "epilogue tensor shape != output shape (and not one broadcast row)";
    }
    return {};
}

// u2 weight-only-quantized fully connected layer on XMX/DPAS: woq_u2_gemm_dual.cm (tiled) for M > 8 and
// woq_u2_gemm_dual_gemv.cm for M <= 8, both built with WLAYOUT=1 (N-major weights and scales / zero points).
//
// Selected only for: Xe2 or Xe3 with CM JIT support, >= 96 KB SLM; f16 activations without padding; u2
// weights [N, K] (weights_transposed); f16 decompression scales [N, K/64] bfyx (group size 64); u8
// decompression zero points [N, K/64] bfyx (required: the kernels have no scalar / f16 / absent zero-point
// path); no dynamically quantized activations; f16 or f32 output (OUT_F16); K % 64 == 0 and N % 32 == 0;
// an epilogue woq_u2_epilogue supports (none, bias, fused add, fused SwiGLU) whose tensor passes
// woq_u2_check_epi_operand (f16 / f32; shaped like the output -- a bias / per-column operand [N] therefore
// only with M == 1). M (the product of the leading
// dims) may be dynamic. Everything else falls back to the other fully_connected implementations (for u2:
// the OCL reference kernel). prepare_quantization keeps per-group scales / zero points of u2 FCs in bfyx for
// this kernel, and prepare_primitive_fusing lets u2 FCs that pass validate_base take post-op fusions.
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
        return validate_base(node, true);
    }

    // Everything but the epilogue: whether this FC could run on the CM kernels if its bias / fused post-ops
    // are supported. prepare_primitive_fusing uses it to let such FCs take post-op fusions.
    static bool accepts_post_op_fusion(const program_node& node) {
        return validate_base(node, false);
    }

    static bool validate_base(const program_node& node, bool check_epilogue) {
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

        const auto& fc_node = node.as<fully_connected>();
        const auto& prim = fc_node.get_primitive();
        if (!prim->compressed_weights)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights not compressed");
        if (!prim->decompression_scale.is_valid())
            CM_FC_LOG_AND_RETURN_FALSE(node, "no decompression scale");
        if (prim->dynamic_quantized_activation)
            CM_FC_LOG_AND_RETURN_FALSE(node, "dynamic quantized activation not supported");
        if (!prim->weights_transposed)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights not transposed");
        if (prim->weights_rank != 2)
            CM_FC_LOG_AND_RETURN_FALSE(node, "weights_rank != 2: " << prim->weights_rank);

        const bool has_bias = fc_node.bias_term();
        const auto in_layouts = node.get_input_layouts();
        if (in_layouts.size() <= woq_u2_scale_idx(has_bias))
            CM_FC_LOG_AND_RETURN_FALSE(node, "missing decompression scale input");
        const auto& in = in_layouts[0];
        const auto& wei = in_layouts[1];
        const auto& scale = in_layouts[woq_u2_scale_idx(has_bias)];
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
        if (!prim->decompression_zero_point.is_valid() || in_layouts.size() <= woq_u2_zp_idx(has_bias))
            CM_FC_LOG_AND_RETURN_FALSE(node, "no decompression zero point");
        const auto& zp = in_layouts[woq_u2_zp_idx(has_bias)];
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

        if (!check_epilogue)
            return true;

        // Epilogue: bias / fused post-ops the kernels can apply, and their tensor.
        const auto epi = woq_u2_epilogue(node.get_fused_primitives(), has_bias);
        if (!epi.ok())
            CM_FC_LOG_AND_RETURN_FALSE(node, epi.error);
        if (epi.dep >= 0) {
            if (static_cast<size_t>(epi.dep) >= in_layouts.size())
                CM_FC_LOG_AND_RETURN_FALSE(node, "epilogue tensor dependency missing");
            const auto why = woq_u2_check_epi_operand(in_layouts[static_cast<size_t>(epi.dep)], out, N);
            if (!why.empty())
                CM_FC_LOG_AND_RETURN_FALSE(node, why);
        }
        return true;
    }
};

}  // namespace ov::intel_gpu::cm
