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

// Weight matrices of the routed experts in input order (gate, up, down); the shared expert's follow at SHARED_GATE_WEIGHT
constexpr size_t INPUTS_PER_MATRIX = 3;

size_t weight_input(size_t matrix, bool shared) {
    const auto first = shared ? MOE3GemmInputIndex::SHARED_GATE_WEIGHT : MOE3GemmInputIndex::WEIGHT_0;
    return static_cast<size_t>(first) + matrix * INPUTS_PER_MATRIX;
}

struct Int3MatrixDesc {
    size_t k;
    size_t n;
    size_t weight_group_size;  // 0 if the weights or scales don't suit the kernel
    bool zp_scalar;
    ov::element::Type zp_type;
};

// matrix: 0 gate, 1 up, 2 down
Int3MatrixDesc get_int3_matrix_desc(const RuntimeParams& params, size_t matrix, bool shared) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const bool down = matrix == 2;
    const size_t k = down ? config.inter_size : config.hidden_size;
    const size_t n = down ? config.hidden_size : config.inter_size;
    const size_t experts = shared ? 1 : config.num_expert;
    const auto& weight = params.get_input_layout(weight_input(matrix, shared));
    const auto& scale = params.get_input_layout(weight_input(matrix, shared) + 1);
    Int3MatrixDesc desc{k, n, 0, false, ov::element::dynamic};
    if (weight.data_type != ov::element::u3 || scale.data_type != ov::element::f16) {
        return desc;
    }
    const size_t weight_groups = scale.count() / (experts * n);
    if (weight_groups == 0 || scale.count() != experts * n * weight_groups || k % weight_groups != 0) {
        return desc;
    }
    // The zero point is shared by all the experts, or has the scales' shape
    if (config.has_zp) {
        const auto& zp = params.get_input_layout(weight_input(matrix, shared) + 2);
        desc.zp_scalar = zp.count() == 1;
        desc.zp_type = zp.data_type;
        if (!one_of(zp.data_type, {ov::element::u8, ov::element::i8}) || (!desc.zp_scalar && zp.count() != scale.count())) {
            return desc;
        }
    }
    desc.weight_group_size = k / weight_groups;
    return desc;
}

}  // namespace

bool MoE3GemmInt3Gemv::has_shared_expert(const RuntimeParams& params) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    return config.num_shared_expert > 0 && params.input_layouts.size() > static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_GATE_WEIGHT);
}

bool MoE3GemmInt3Gemv::is_supported(const RuntimeParams& params) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    if (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0)).data_type != ov::element::u3 ||
        params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)).data_type != ov::element::f16) {
        return false;
    }
    // A shared expert of another precision runs the other weight types' kernels
    const bool shared = has_shared_expert(params);
    if (config.num_shared_expert > 1 || (config.num_shared_expert == 1 && !shared)) {
        return false;
    }
    if (shared && params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::SHARED_GATE_GATE_WEIGHT)).data_type != ov::element::f16) {
        return false;
    }
    // The kernel takes one group size and zero point kind per projection: gate and up, and the shared expert's
    // matrices, must match the routed gate (or down) matrix
    for (size_t matrix = 0; matrix < 3; ++matrix) {
        const auto ref = get_int3_matrix_desc(params, matrix == 2 ? 2 : 0, false);
        const auto proj = matrix == 2 ? Projection::DOWN : Projection::GATE_UP;
        if (ref.k % (SIMD * 32) != 0 || ref.weight_group_size == 0 || ref.weight_group_size % 32 != 0 || ref.n % (SGN * get_nb(params, proj)) != 0) {
            return false;
        }
        for (bool is_shared : {false, true}) {
            if (is_shared && !shared) {
                continue;
            }
            const auto desc = get_int3_matrix_desc(params, matrix, is_shared);
            if (desc.weight_group_size != ref.weight_group_size || desc.zp_scalar != ref.zp_scalar || desc.zp_type != ref.zp_type) {
                return false;
            }
        }
    }
    return true;
}

// One token runs top_k * n / nb subgroups. The column counts (more for down, whose K is short) suit a GPU with 512
// hardware threads; on larger GPUs they shrink until a single token keeps at least half of the threads busy.
size_t MoE3GemmInt3Gemv::get_nb(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto& info = params.get_device_info();
    const size_t threads = static_cast<size_t>(info.execution_units_count) * std::max<uint32_t>(info.num_threads_per_eu, 1);
    const size_t n = proj == Projection::DOWN ? config.hidden_size : config.inter_size;
    size_t nb = proj == Projection::DOWN ? 32 : 8;
    while (nb > 1 && config.top_k * n / nb < threads / 2) {
        nb /= 2;
    }
    return nb;
}

// Most tokens per step for which the u3 batched GEMV beats the OneDNN grouped GEMM on a Qwen3.6-35B-A3B MoE layer. The
// GEMV is bound by the memory bandwidth, the grouped GEMM by the compute: on the shared memory of an integrated GPU the
// crossover drops with the EU count (6 tokens on LNL 258V with 64 EUs, about 2 on PTL 386H with 96 EUs), while a
// discrete GPU stays on the GEMV much longer (about 16 tokens on BMG B570).
size_t MoE3GemmInt3Gemv::get_default_threshold(const RuntimeParams& params) {
    const auto& info = params.get_device_info();
    if (info.dev_type == cldnn::device_type::discrete_gpu) {
        return 16;
    }
    return info.execution_units_count <= 64 ? 6 : 2;
}

JitConstants MoE3GemmInt3Gemv::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto desc = get_int3_matrix_desc(params, m_proj == Projection::DOWN ? 2 : 0, false);
    jit.make("GEMV_DOWN", m_proj == Projection::DOWN ? 1 : 0);
    jit.make("K_SIZE", desc.k);
    jit.make("N_SIZE", desc.n);
    jit.make("WEI_GROUP_SIZE", desc.weight_group_size);
    jit.make("HAS_ZP", config.has_zp ? 1 : 0);
    if (config.has_zp) {
        jit.make("ZP_SCALAR", desc.zp_scalar ? 1 : 0);
        jit.make("ZP_TYPE", desc.zp_type == ov::element::i8 ? "char" : "uchar");
    }
    jit.make("MAX_TOPK", config.top_k);
    jit.make("SHARED_EXPERT", has_shared_expert(params) ? 1 : 0);
    jit.make("SGN", SGN);
    jit.make("NB", get_nb(params, m_proj));
    switch (config.activation_type) {
    case ov::op::internal::MOE::Activation_type::SWIGLU:
        break;
    case ov::op::internal::MOE::Activation_type::GEGLU_TANH:
        jit.make("GATE_ACT_GELU_TANH", 1);
        break;
    case ov::op::internal::MOE::Activation_type::GEGLU_ERF:
        jit.make("GATE_ACT_GELU_ERF", 1);
        break;
    default:
        OPENVINO_THROW("MoE u3 batched GEMV: unsupported gate activation ", static_cast<int>(config.activation_type));
    }
    return jit;
}

}  // namespace ov::intel_gpu::ocl
