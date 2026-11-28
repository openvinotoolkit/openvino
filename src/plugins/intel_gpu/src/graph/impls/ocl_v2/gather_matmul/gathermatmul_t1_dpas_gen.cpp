// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gathermatmul_t1_dpas_gen.hpp"

#include <cstdlib>

#include "gather_matmul_gen_micro.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/gather_matmul.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned
namespace ov::intel_gpu::ocl {

namespace {
constexpr size_t kWgSubgroups = 8;  // WG_SG in the kernel
constexpr size_t kChunkShorts = 8 * 16;

size_t scale_groups(const RuntimeParams& params) {
    const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
    const auto& scale_layout = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT_SCALE);
    return scale_layout.count() / (weight_shape[0] * weight_shape[1]);
}

// unit = values decoded per step: t1 256 (2 groups), u1 128
enum class Fmt { T1, U1 };
Fmt fmt_of(const gather_matmul& d) {
    return d.weights_u1 ? Fmt::U1 : Fmt::T1;
}
size_t unit_values(Fmt f) {
    return f == Fmt::T1 ? 256 : 128;
}
size_t chunks(Fmt f) {
    return f == Fmt::T1 ? 10 : 4;
}
size_t subt(Fmt f) {
    return f == Fmt::U1 ? 4 : 2;
}
size_t round_up(size_t v, size_t m) {
    return (v + m - 1) / m * m;
}
}  // namespace

bool gathermatmul_t1_dpas_supported(const RuntimeParams& params) {
    const auto desc = params.typed_desc<gather_matmul>();
    if (!desc->weights_t1 && !desc->weights_u1) {
        return false;
    }
    const auto k = static_cast<size_t>(desc->t1_k);
    const size_t groups = scale_groups(params);
    return groups > 0 && k == groups * 128 && k % unit_values(fmt_of(*desc)) == 0;
}

size_t gathermatmul_t1_dpas_min_tokens() {
    return 16;
}

size_t gathermatmul_t1_dpas_max_subtiles(size_t rows, size_t n_experts) {
    return (rows + 7) / 8 + n_experts;
}

size_t gathermatmul_t1_dpas_xd_elems(const RuntimeParams& params, size_t max_subtiles) {
    const auto desc = params.typed_desc<gather_matmul>();
    const auto f = fmt_of(*desc);
    const size_t units = static_cast<size_t>(desc->t1_k) / unit_values(f);
    return max_subtiles * units * chunks(f) * kChunkShorts;
}

JitConstants GatherMatmulT1DpasGenerator::get_jit_constants(const RuntimeParams& params) const {
    auto jit = make_base_jit_constants(params);
    const auto desc = params.typed_desc<gather_matmul>();
    const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
    jit.make("STAGE", m_stage);
    jit.make("K_SIZE", static_cast<size_t>(desc->t1_k));
    jit.make("N_SIZE", weight_shape[1]);
    jit.make("N_EXPERTS", weight_shape[0]);
    jit.make("SUBT", subt(fmt_of(*desc)));
    if (desc->scales_fp8) {
        jit.make("SCALES_FP8", 1);
    }
    if (desc->weights_u1) {
        jit.make("WEIGHTS_U1", 1);
        jit.make("U1_ZP", std::to_string(desc->u1_zp) + "f");
    }
    return jit;
}

Arguments GatherMatmulT1DpasGenerator::get_arguments_desc(const RuntimeParams& params) const {
    Arguments args;
    auto ib = [&](uint32_t idx) {
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, idx});
    };
    auto in = [&](uint32_t idx) {
        args.push_back({ArgumentDescriptor::Types::INPUT, idx});
    };
    auto scalar = [&](uint32_t idx) {
        args.push_back({ArgumentDescriptor::Types::SCALAR, idx});
    };
    switch (m_stage) {
    case 1:  // zero
        ib(T1D_COUNTS);
        break;
    case 2:  // count
        in(gather_matmul::BGMInputIdx::INDICES);
        ib(T1D_COUNTS);
        ib(T1D_POS);
        scalar(0);  // rows
        break;
    case 3:  // build
        ib(T1D_COUNTS);
        ib(T1D_OFFSETS);
        ib(T1D_SUBTILES);
        ib(T1D_TILES);
        ib(T1D_META);
        break;
    case 4:  // place
        in(gather_matmul::BGMInputIdx::INDICES);
        ib(T1D_OFFSETS);
        ib(T1D_POS);
        ib(T1D_ROW_TOKEN);
        ib(T1D_ROW_SLOT);
        scalar(0);  // rows
        scalar(1);  // top_k
        break;
    case 5:  // quant_scale
        in(gather_matmul::BGMInputIdx::INPUT);
        ib(T1D_ROW_TOKEN);
        ib(T1D_ROW_SLOT);
        ib(T1D_XS);
        ib(T1D_XSUM);
        scalar(0);  // rows
        scalar(1);  // n_tokens
        scalar(2);  // n_act
        break;
    case 6:  // quant_dpas
        in(gather_matmul::BGMInputIdx::INPUT);
        ib(T1D_ROW_TOKEN);
        ib(T1D_ROW_SLOT);
        ib(T1D_XS);
        ib(T1D_SUBTILES);
        ib(T1D_META);
        ib(T1D_XD);
        scalar(0);  // n_tokens
        scalar(1);  // n_act
        break;
    default:  // tile
        ib(T1D_XD);
        ib(T1D_XS);
        ib(T1D_XSUM);
        in(gather_matmul::BGMInputIdx::WEIGHT);
        in(gather_matmul::BGMInputIdx::WEIGHT_SCALE);
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        ib(T1D_ROW_TOKEN);
        ib(T1D_ROW_SLOT);
        ib(T1D_TILES);
        ib(T1D_SUBTILES);
        ib(T1D_META);
        scalar(0);  // n_tokens
        break;
    }
    return args;
}

DispatchDataFunc GatherMatmulT1DpasGenerator::get_dispatch_data_func() const {
    return DispatchDataFunc{[stage = m_stage](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {
        auto* rtp = static_cast<GatherMatmulRuntimeParams*>(rt_params);
        auto& wgs = kd.params.workGroups;
        auto& scalars = kd.params.scalars;
        scalars.clear();
        const auto desc = params.typed_desc<gather_matmul>();
        const auto u1 = fmt_of(*desc);
        const size_t k = static_cast<size_t>(desc->t1_k);
        const auto& weight_shape = params.get_input_layout(gather_matmul::BGMInputIdx::WEIGHT).get_shape();
        const size_t n = weight_shape[1], n_experts = weight_shape[0];
        const size_t rows = static_cast<size_t>(rtp->n_tokens) * static_cast<size_t>(rtp->top_k);
        const size_t max_sub = gathermatmul_t1_dpas_max_subtiles(rows, n_experts);
        auto push = [&](int32_t v) {
            ScalarDescriptor s{ScalarDescriptor::Types::INT32};
            s.v.s32 = v;
            scalars.push_back(s);
        };
        switch (stage) {
        case 1:
            wgs.global = {round_up(n_experts, 64), 1, 1};
            wgs.local = {64, 1, 1};
            break;
        case 2:
            wgs.global = {round_up(rows, 64), 1, 1};
            wgs.local = {64, 1, 1};
            push(static_cast<int32_t>(rows));
            break;
        case 3:
            wgs.global = {1, 1, 1};
            wgs.local = {1, 1, 1};
            break;
        case 4:
            wgs.global = {round_up(rows, 64), 1, 1};
            wgs.local = {64, 1, 1};
            push(static_cast<int32_t>(rows));
            push(rtp->top_k);
            break;
        case 5:
            wgs.global = {rows * (k / 128) * 16, 1, 1};
            wgs.local = {16, 1, 1};
            push(static_cast<int32_t>(rows));
            push(rtp->n_tokens);
            push(rtp->n_activated_experts);
            break;
        case 6:
            wgs.global = {max_sub * (k / unit_values(u1)) * chunks(u1) * kChunkShorts, 1, 1};
            wgs.local = {kChunkShorts, 1, 1};
            push(rtp->n_tokens);
            push(rtp->n_activated_experts);
            break;
        default:
            wgs.global = {kWgSubgroups * 16, (n + kWgSubgroups * 16 - 1) / (kWgSubgroups * 16), max_sub};
            wgs.local = {kWgSubgroups * 16, 1, 1};
            push(rtp->n_tokens);
            break;
        }
    }};
}

}  // namespace ov::intel_gpu::ocl
