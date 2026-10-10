// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "activation_inst.h"
#include "data_inst.h"
#include "eltwise_inst.h"
#include "fully_connected_inst.h"
#include "intel_gpu/graph/fused_primitive_desc.hpp"
#include "openvino/util/env_util.hpp"
#include "registry/implementation_manager.hpp"

namespace cldnn {
namespace ocl {

// Ternary weights are 2 bits per value in groups of 128 along K, one f16 scale per group.
constexpr size_t kTernoclGroupSize = 128;
constexpr size_t kTernoclPackFactor = 16;

// OV_TERNOCL_INT2_FOLD_GATES=bias|sigmoid|none selects which GatedDeltaNet gate
// epilogues are folded; bit 0 is the bias, bit 1 the sigmoid.
inline int ternocl_int2_fold_gates() {
    static const int v = [] {
        const char* e = std::getenv("OV_TERNOCL_INT2_FOLD_GATES");
        if (e == nullptr)
            return 3;
        const std::string s(e);
        if (s == "none" || s == "0")
            return 0;
        if (s == "bias")
            return 1;
        if (s == "sigmoid")
            return 2;
        return 3;
    }();
    return v;
}

// Epilogue the kernel runs for a fused chain: 0 none, 1 silu(acc)*other,
// 2 acc+other, 3 acc+bias, 4 sigmoid(acc); -1 when there is no single-pass
// equivalent. `other_dep` receives the dependency index of the eltwise operand.
inline int ternocl_int2_postop(const std::vector<fused_primitive_desc>& fused, bool has_bias, size_t* other_dep = nullptr) {
    const int fold_gates = ternocl_int2_fold_gates();
    if (fused.empty())
        return has_bias ? ((fold_gates & 1) ? 3 : -1) : 0;
    if (has_bias)
        return -1;
    const auto act0 = std::dynamic_pointer_cast<const activation>(fused[0].desc);
    const auto elt_last = std::dynamic_pointer_cast<const eltwise>(fused.back().desc);
    if (elt_last && other_dep != nullptr)
        *other_dep = static_cast<size_t>(fused.back().outer_dep_start_idx);
    if (fused.size() == 2 && act0 && act0->activation_function == activation_func::swish && elt_last && elt_last->mode == eltwise_mode::prod)
        return 1;
    if (fused.size() == 1 && elt_last && elt_last->mode == eltwise_mode::sum)
        return 2;
    if (fused.size() == 1 && act0 && act0->activation_function == activation_func::logistic && (fold_gates & 2))
        return 4;
    return -1;
}

// Weight-only-quantized FullyConnected on 2-bit ternary weights, executed by the
// TernOCL int2 x f16 up-convert OpenCL kernels (fp16 DPAS). OpenVINO has no
// signed 2-bit type, so ternary weights arrive as u2 codes {0,1,2} plus a zero
// point of 1; they are re-encoded to the kernel's {0,1,3} and packed once, when
// the impl is created.
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
        const bool dbg = ov::util::getenv_bool("OV_TERNOCL_INT2_DEBUG");
        if (ov::util::getenv_bool("OV_TERNOCL_INT2_DISABLE"))
            return false;
#define TERNOCL_REJECT(reason)                                                                           \
    do {                                                                                                 \
        if (dbg)                                                                                         \
            std::cerr << "[ternocl-int2] reject: " << (reason) << " node=" << fc_node.id() << std::endl; \
        /* a fused Hadamard input exists only here; another impl would ignore it */                      \
        OPENVINO_ASSERT(fc_prim->hadamard_block == 0,                                                    \
                        "[GPU] ternocl int2: ",                                                          \
                        fc_node.id(),                                                                    \
                        " carries a Hadamard input transform but the TernOCL impl rejected it: ",        \
                        (reason));                                                                       \
        return false;                                                                                    \
    } while (0)

        if (!fc_prim->compressed_weights)
            TERNOCL_REJECT("not compressed_weights");
        if (fc_node.weights().get_output_layout(false).data_type != data_types::u2)
            TERNOCL_REJECT("weights are not u2");
        if (!fc_node.weights().is_type<data>())
            TERNOCL_REJECT("weights are not a constant");

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

        // A constant bias folds into the epilogue (POSTOP 3); it has the output type.
        if (fc_prim->bias.is_valid()) {
            if (!fc_node.bias().is_type<data>())
                TERNOCL_REJECT("bias is not a constant");
            if (fc_node.bias().get_output_layout(false).data_type != out_layout.data_type)
                TERNOCL_REJECT("bias type differs from the output type");
        }
        size_t other_dep = 0;
        const int postop = ternocl_int2_postop(fc_node.get_fused_primitives(), fc_prim->bias.is_valid(), &other_dep);
        if (postop < 0)
            TERNOCL_REJECT("fused chain has no folded epilogue");
        if ((postop == 1 || postop == 2) &&
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

        // OV_TERNOCL_INT2_ONLY_N=6144,24576 restricts the impl to the listed output widths (bisecting aid).
        if (const char* only = std::getenv("OV_TERNOCL_INT2_ONLY_N")) {
            const std::string want = "," + std::string(only) + ",";
            if (want.find("," + std::to_string(N) + ",") == std::string::npos)
                TERNOCL_REJECT("N not in OV_TERNOCL_INT2_ONLY_N");
        }

        if (dbg)
            std::cerr << "[ternocl-int2] accepted " << fc_node.id() << std::endl;
        return true;
#undef TERNOCL_REJECT
    }
};

}  // namespace ocl
}  // namespace cldnn
