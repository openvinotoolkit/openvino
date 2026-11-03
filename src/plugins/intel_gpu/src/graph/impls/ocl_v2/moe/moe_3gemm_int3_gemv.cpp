// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_int3_gemv.hpp"

#include <algorithm>

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

namespace {

struct Int3ProjectionShape {
    size_t k;
    size_t n;
    size_t weight_group_size;  // 0 if the scales don't split K into equal groups
};

Int3ProjectionShape get_int3_projection_shape(const RuntimeParams& params, Int3Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const bool down = proj == Int3Projection::DOWN;
    const size_t k = down ? config.inter_size : config.hidden_size;
    const size_t n = down ? config.hidden_size : config.inter_size;
    const auto scale_idx = static_cast<size_t>(down ? MOE3GemmInputIndex::SCALE_2 : MOE3GemmInputIndex::SCALE_0);
    const size_t weight_groups = params.get_input_layout(scale_idx).count() / (config.num_expert * n);
    const size_t weight_group_size = weight_groups > 0 && k % weight_groups == 0 ? k / weight_groups : 0;
    return {k, n, weight_group_size};
}

}  // namespace

bool int3_gemv_supported(const RuntimeParams& params) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    // The shared expert is only fused into the batched GEMV kernels of the other weight types
    if (config.num_shared_expert != 0 || params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)).data_type != ov::element::f16) {
        return false;
    }
    for (auto idx : {MOE3GemmInputIndex::SCALE_0, MOE3GemmInputIndex::SCALE_1, MOE3GemmInputIndex::SCALE_2}) {
        if (params.get_input_layout(static_cast<size_t>(idx)).data_type != ov::element::f16) {
            return false;
        }
    }
    if (config.has_zp) {
        for (auto idx : {MOE3GemmInputIndex::ZP_0, MOE3GemmInputIndex::ZP_1, MOE3GemmInputIndex::ZP_2}) {
            if (!one_of(params.get_input_layout(static_cast<size_t>(idx)).data_type, {ov::element::u8, ov::element::i8})) {
                return false;
            }
        }
    }
    for (auto proj : {Int3Projection::GATE_UP, Int3Projection::DOWN}) {
        const auto shape = get_int3_projection_shape(params, proj);
        if (shape.k % (INT3_GEMV_SIMD * 32) != 0 || shape.weight_group_size == 0 || shape.weight_group_size % 32 != 0 ||
            shape.n % (INT3_GEMV_SGN * int3_gemv_nb(params, proj)) != 0) {
            return false;
        }
    }
    return true;
}

// One token runs top_k * n / nb subgroups. The column counts (more for down, whose K is short) suit a GPU with 512
// hardware threads; on larger GPUs they shrink until a single token keeps at least half of the threads busy.
size_t int3_gemv_nb(const RuntimeParams& params, Int3Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto& info = params.get_device_info();
    const size_t threads = static_cast<size_t>(info.execution_units_count) * std::max<uint32_t>(info.num_threads_per_eu, 1);
    const size_t n = proj == Int3Projection::DOWN ? config.hidden_size : config.inter_size;
    size_t nb = proj == Int3Projection::DOWN ? 32 : 8;
    while (nb > 1 && config.top_k * n / nb < threads / 2) {
        nb /= 2;
    }
    return nb;
}

// Most tokens per step for which the u3 batched GEMV beats the OneDNN grouped GEMM on a Qwen3.6-35B-A3B MoE layer:
// 6 on LNL (9% faster at 2 tokens, 5% at 3, even at 6, 5% slower at 8). BMG stays on the GEMV much longer (24).
size_t int3_gemv_threshold(const RuntimeParams& params) {
    const size_t configured = params.get_program().get_config().get_moe_u3_batched_gemv_threshold();
    if (configured > 0) {
        return configured;
    }
    return params.get_device_info().dev_type == cldnn::device_type::discrete_gpu ? 24 : 6;
}

JitConstants MoE3GemmInt3Gemv::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto shape = get_int3_projection_shape(params, m_proj);
    jit.make("GEMV_DOWN", m_proj == Int3Projection::DOWN ? 1 : 0);
    jit.make("K_SIZE", shape.k);
    jit.make("N_SIZE", shape.n);
    jit.make("WEI_GROUP_SIZE", shape.weight_group_size);
    jit.make("HAS_ZP", config.has_zp ? 1 : 0);
    if (config.has_zp) {
        const auto zp_idx = static_cast<size_t>(m_proj == Int3Projection::DOWN ? MOE3GemmInputIndex::ZP_2 : MOE3GemmInputIndex::ZP_0);
        const auto& zp = params.get_input_layout(zp_idx);
        jit.make("ZP_SCALAR", zp.count() == 1 ? 1 : 0);
        jit.make("ZP_TYPE", zp.data_type == ov::element::i8 ? "char" : "uchar");
    }
    jit.make("MAX_TOPK", config.top_k);
    jit.make("SGN", INT3_GEMV_SGN);
    jit.make("NB", int3_gemv_nb(params, m_proj));
    if (config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        jit.make("GATE_ACT_GELU_TANH", 1);
    } else if (config.activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        jit.make("GATE_ACT_GELU_ERF", 1);
    }
    return jit;
}

}  // namespace ov::intel_gpu::ocl
