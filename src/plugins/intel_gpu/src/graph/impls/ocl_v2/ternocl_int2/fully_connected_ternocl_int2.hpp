// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "activation_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

// Ternary weights are 2 bits per value in groups of 128 along K, one f16 scale per group.
inline constexpr size_t kTernoclGroupSize = 128;
inline constexpr size_t kTernoclPackFactor = 16;

// Epilogue folded into the kernel; the value is passed to it as the POSTOP macro.
enum class TernoclPostop : int {
    none = 0,
    silu_mul = 1,      // silu(acc) * other (gate/up)
    residual_add = 2,  // acc + other
    bias = 3,          // acc + bias
    sigmoid = 4,       // sigmoid(acc)
};

inline bool ternocl_postop_reads_other(TernoclPostop p) {
    return p == TernoclPostop::silu_mul || p == TernoclPostop::residual_add;
}

// The epilogue for a fused chain, or nullopt when it has no single-pass equivalent.
// `other_dep` receives the dependency index of the eltwise operand.
inline std::optional<TernoclPostop> ternocl_int2_postop(const std::vector<fused_primitive_desc>& fused, bool has_bias, size_t* other_dep = nullptr) {
    if (fused.empty())
        return has_bias ? TernoclPostop::bias : TernoclPostop::none;
    if (has_bias)
        return std::nullopt;
    const auto act0 = std::dynamic_pointer_cast<const activation>(fused[0].desc);
    const auto elt_last = std::dynamic_pointer_cast<const eltwise>(fused.back().desc);
    if (elt_last && other_dep != nullptr)
        *other_dep = static_cast<size_t>(fused.back().outer_dep_start_idx);
    if (fused.size() == 2 && act0 && act0->activation_function == activation_func::swish && elt_last && elt_last->mode == eltwise_mode::prod)
        return TernoclPostop::silu_mul;
    if (fused.size() == 1 && elt_last && elt_last->mode == eltwise_mode::sum)
        return TernoclPostop::residual_add;
    if (fused.size() == 1 && act0 && act0->activation_function == activation_func::logistic)
        return TernoclPostop::sigmoid;
    return std::nullopt;
}

// The kernels read the grouped f16 scales as [groups, N]. With transpose_b_scale the scales are
// [N, groups], which prepare_quantization reorders to fbyx, i.e. that physical layout (a single
// group is not reordered: [N, 1] in bfyx is the same); otherwise they are [groups, N] in bfyx.
inline bool ternocl_scales_group_major(const layout& l, bool transpose_b_scale, int64_t N, int64_t G) {
    if (l.data_type != data_types::f16 || l.data_padding || l.is_dynamic())
        return false;
    const auto& ps = l.get_partial_shape();
    if (ps.size() < 2)
        return false;
    for (size_t i = 2; i < ps.size(); ++i) {
        if (ps[i].get_length() != 1)
            return false;
    }
    const auto d0 = ps[0].get_length();
    const auto d1 = ps[1].get_length();
    if (transpose_b_scale)
        return d0 == N && d1 == G && (l.format == format::fbyx || (G == 1 && l.format == format::bfyx));
    return d0 == G && d1 == N && l.format == format::bfyx;
}

// Weight-only-quantized FullyConnected on 2-bit ternary weights, executed by the
// TernOCL int2 x f16 up-convert OpenCL kernels (fp16 DPAS). OpenVINO has no
// signed 2-bit type, so ternary weights arrive as u2 codes {0,1,2} plus a zero
// point of 1. The impl requests format::is_os_yx_isv16_ternary through its weights
// reorder params: the weights are re-encoded to the kernel's {0,1,3} and packed by
// TernoclInt2WeightsReorderImplementationManager when the program folds its constants.
struct TernoclInt2FCImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("TernoclInt2FCImplementationManager")
    TernoclInt2FCImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::ocl, shape_type, vf) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    std::unique_ptr<primitive_impl> create_impl(const kernel_impl_params& params) const override;
    in_out_fmts_t query_formats(const program_node& node) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<fully_connected>());
        const auto& fc_node = node.as<fully_connected>();
        const auto& fc_prim = fc_node.get_primitive();
        if (!node.get_program().get_config().get_enable_ternocl_int2())
            return false;
        // The kernels use SIMD16 DPAS and 2D block loads.
        const auto& device_info = node.get_program().get_engine().get_device_info();
        if (device_info.arch < gpu_arch::xe2 || !device_info.supports_immad)
            return false;
#define TERNOCL_REJECT(reason)                                                                               \
    do {                                                                                                     \
        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] reject " << fc_node.id() << ": " << (reason) << std::endl; \
        /* no other impl accepts a fused Hadamard input: fail here with the reason */                        \
        OPENVINO_ASSERT(fc_prim->hadamard_block == 0,                                                        \
                        "[GPU] ternocl int2: ",                                                              \
                        fc_node.id(),                                                                        \
                        " carries a Hadamard input transform but the TernOCL impl rejected it: ",            \
                        (reason));                                                                           \
        return false;                                                                                        \
    } while (0)

        if (!fc_prim->compressed_weights)
            TERNOCL_REJECT("not compressed_weights");
        if (fc_node.weights().get_output_layout(false).data_type != data_types::u2)
            TERNOCL_REJECT("weights are not u2");
        if (!fc_node.weights().is_type<data>())
            TERNOCL_REJECT("weights are not a constant");
        // Ternary {-1, 0, +1} arrives as u2 codes {0, 1, 2} with a scalar zero point of 1.
        if (!fc_prim->decompression_zero_point_scalar.has_value() || std::lround(fc_prim->decompression_zero_point_scalar.value()) != 1)
            TERNOCL_REJECT("zero point is not the scalar 1");

        const auto& in_layout = fc_node.get_input_layout(0);
        const auto& out_layout = fc_node.get_output_layout(0);
        if (in_layout.data_type != data_types::f16)
            TERNOCL_REJECT("activations are not f16");
        if (out_layout.data_type != data_types::f16 && out_layout.data_type != data_types::f32)
            TERNOCL_REJECT("output is not f16 or f32");
        if (in_layout.format != format::bfyx || out_layout.format != format::bfyx)
            TERNOCL_REJECT("unsupported activation format");
        if (in_layout.data_padding || out_layout.data_padding)
            TERNOCL_REJECT("padded activations");

        // A constant bias folds into the epilogue (TernoclPostop::bias); it has the output type.
        if (fc_prim->bias.is_valid()) {
            if (!fc_node.bias().is_type<data>())
                TERNOCL_REJECT("bias is not a constant");
            if (fc_node.bias().get_output_layout(false).data_type != out_layout.data_type)
                TERNOCL_REJECT("bias type differs from the output type");
        }
        size_t other_dep = 0;
        const auto postop = ternocl_int2_postop(fc_node.get_fused_primitives(), fc_prim->bias.is_valid(), &other_dep);
        if (!postop)
            TERNOCL_REJECT("fused chain has no folded epilogue");
        if (ternocl_postop_reads_other(*postop) &&
            (other_dep >= fc_node.get_dependencies().size() || fc_node.get_dependency(other_dep).get_output_layout(false).data_type != data_types::f16))
            TERNOCL_REJECT("eltwise operand is not f16");

        // Weight layouts are canonicalized to 4D, so [N, K] arrives as [N, K, 1, 1].
        const auto& wei_pshape = fc_node.weights().get_output_layout(false).get_partial_shape();
        if (wei_pshape.is_dynamic() || wei_pshape.size() < 2)
            TERNOCL_REJECT("weights shape is not a static [N, K]: " + wei_pshape.to_string());
        for (size_t i = 2; i < wei_pshape.size(); ++i) {
            if (wei_pshape[i].get_length() != 1)
                TERNOCL_REJECT("weights shape is not a static [N, K]: " + wei_pshape.to_string());
        }
        const auto N = wei_pshape[0].get_length();
        const auto K = wei_pshape[1].get_length();
        if (K % static_cast<int64_t>(kTernoclGroupSize) != 0)
            TERNOCL_REJECT("K is not a multiple of the group size");
        // One lane per output column in 16-wide sub-group blocks; the tail is clipped, not padded.
        if (N % 16 != 0)
            TERNOCL_REJECT("N is not a multiple of 16");
        if (fc_prim->hadamard_block != 0 && (fc_prim->hadamard_block != 1024 || K % 1024 != 0))
            TERNOCL_REJECT("hadamard block must be 1024 and divide K");
        if (fc_prim->hadamard_block != 0 && !fc_prim->hadamard_signs.empty() && fc_prim->hadamard_signs.size() != static_cast<size_t>(K))
            TERNOCL_REJECT("hadamard signs length differs from K");

        const size_t scale_dep = fc_prim->bias.is_valid() ? 3 : 2;
        if (!fc_prim->decompression_scale.is_valid() || scale_dep >= fc_node.get_dependencies().size())
            TERNOCL_REJECT("no decompression scale");
        if (!ternocl_scales_group_major(fc_node.get_dependency(scale_dep).get_output_layout(false),
                                        fc_prim->transpose_b_scale,
                                        N,
                                        K / static_cast<int64_t>(kTernoclGroupSize)))
            TERNOCL_REJECT("decompression scales are not f16 [groups, N]: " + fc_node.get_dependency(scale_dep).get_output_layout(false).to_short_string());

        GPU_DEBUG_TRACE_DETAIL << "[ternocl-int2] accept " << fc_node.id() << std::endl;
        return true;
#undef TERNOCL_REJECT
    }
};

}  // namespace ocl
}  // namespace cldnn
