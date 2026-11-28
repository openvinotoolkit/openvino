// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul_t1_gen.hpp"

#include <algorithm>

#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

namespace {
size_t t1_groups(const RuntimeParams& params) {
    const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
    const auto& scale_layout = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT_SCALE);
    return scale_layout.count() / (weight_shape[0] * weight_shape[1]);
}

// dp4a variant (int8 activations in SLM, SWAR digit/bit extraction): needs 64- or 128-value weight groups and, for
// t1, K a multiple of 256 (4-block units). Other shapes take the per-value reference kernel.
constexpr size_t kDp4aSubgroups = 8;
// Output columns a work-group handles per subgroup row slot.
constexpr size_t kDp4aRowIters = 4;

bool use_dp4a(const RuntimeParams& params) {
    const auto desc = params.typed_desc<gather_matmul>();
    const auto k = static_cast<size_t>(desc->t1_k);
    const size_t group = k / t1_groups(params);
    if (group != 64 && group != 128) {
        return false;
    }
    const bool bit_units = desc->weights_u1;  // a unit is one group; ternary: 256 values
    if (bit_units ? (k % group != 0) : (k % 256 != 0)) {
        return false;
    }
    // fewer than 16 units per row are spread over a power-of-two number of lanes (shuffle-xor reduction)
    const size_t units = bit_units ? k / group : k / 256;
    return units >= 16 || (units & (units - 1)) == 0;
}

// output columns covered by one workgroup of the dp4a variant
size_t dp4a_rows_per_wg(const RuntimeParams& params) {
    const auto desc = params.typed_desc<gather_matmul>();
    const auto k = static_cast<size_t>(desc->t1_k);
    const size_t group = k / t1_groups(params);
    const size_t units = desc->weights_u1 ? k / group : k / 256;
    const size_t lanes_per_row = std::min<size_t>(units, 16);
    return kDp4aSubgroups * (16 / lanes_per_row) * kDp4aRowIters;
}
}  // namespace

bool gathermatmul_t1_shared_quant(const RuntimeParams& params) {
    return use_dp4a(params);
}

size_t gathermatmul_t1_shared_xq_uints(const RuntimeParams& params) {
    const auto k = static_cast<size_t>(params.typed_desc<gather_matmul>()->t1_k);
    return (k / 256 + 1) * 70;  // an upper bound of the permuted int8 codes per 256 values over both formats
}

size_t gathermatmul_t1_shared_groups(const RuntimeParams& params) {
    return t1_groups(params);
}

JitConstants GatherMatmulT1Generator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    const auto desc = params.typed_desc<gather_matmul>();
    const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
    const auto k = static_cast<size_t>(desc->t1_k);
    const size_t groups = t1_groups(params);
    OPENVINO_ASSERT(k % 64 == 0 && groups > 0 && k % groups == 0 && (k / groups) % 64 == 0, "GatherMatmul t1: unsupported K ", k, " or scale groups ", groups);
    jit.make("K_SIZE", k);
    jit.make("N_SIZE", weight_shape[1]);
    jit.make("WEI_GROUP_SIZE", k / groups);
    if (use_dp4a(params)) {
        jit.make("T1_DP4A", 1);
        jit.make("WG_SG", kDp4aSubgroups);
        jit.make("ROW_ITERS", kDp4aRowIters);
    }
    if (desc->scales_fp8) {
        jit.make("SCALES_FP8", 1);
    }
    if (desc->weights_u1) {
        jit.make("WEIGHTS_U1", 1);
        jit.make("U1_ZP", std::to_string(desc->u1_zp) + "f");
    }
    if (gathermatmul_t1_shared_quant(params)) {
        jit.make(m_quant_only ? "T1_QUANT_ONLY" : "SHARED_XQ", 1);
    }
    return jit;
}

Arguments GatherMatmulT1Generator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    if (params.is_dynamic()) {
        args.push_back({ArgumentDescriptor::Types::SHAPE_INFO, 0});
    }
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INPUT});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT});
    args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::INDICES});
    args.push_back({ArgumentDescriptor::Types::INPUT, gather_matmul::BGMInputIdx::WEIGHT_SCALE});
    if (gathermatmul_t1_shared_quant(params)) {
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 20});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 21});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, 22});
    }
    args.push_back({ArgumentDescriptor::Types::SCALAR, 0});  // n_tokens
    args.push_back({ArgumentDescriptor::Types::SCALAR, 1});  // n_act
    args.push_back({ArgumentDescriptor::Types::SCALAR, 2});  // top_k
    return args;
}

DispatchDataFunc GatherMatmulT1Generator::get_dispatch_data_func() const {
    return DispatchDataFunc{[quant_only = m_quant_only](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();

        const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
        const size_t n = weight_shape[1];
        const size_t n_tokens = static_cast<size_t>(rtp->n_tokens);
        const size_t top_k = static_cast<size_t>(rtp->top_k);

        if (quant_only) {
            wgs.global = {kDp4aSubgroups * 16, 1, n_tokens * static_cast<size_t>(rtp->n_activated_experts)};
            wgs.local = {kDp4aSubgroups * 16, 1, 1};
        } else if (use_dp4a(params)) {
            const size_t rows = dp4a_rows_per_wg(params);
            wgs.global = {kDp4aSubgroups * 16, (n + rows - 1) / rows, n_tokens * top_k};
            wgs.local = {kDp4aSubgroups * 16, 1, 1};
        } else {
            wgs.global = {16, n, n_tokens * top_k};
            wgs.local = {16, 1, 1};
        }

        for (int32_t v : {rtp->n_tokens, rtp->n_activated_experts, rtp->top_k}) {
            ScalarDescriptor s{ScalarDescriptor::Types::INT32};
            s.v.s32 = v;
            scalars.push_back(s);
        }
    }};
}

}  // namespace ov::intel_gpu::ocl
