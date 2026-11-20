// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"

namespace ov::intel_gpu::ocl {

// Weight matrices of the routed experts in input order (gate, up, down), each followed by its scale and zero point;
// the shared expert's follow at SHARED_GATE_WEIGHT
constexpr size_t MOE_INPUTS_PER_MATRIX = 3;

inline size_t moe_weight_input(size_t matrix, bool shared = false) {
    const auto first = shared ? MOE3GemmInputIndex::SHARED_GATE_WEIGHT : MOE3GemmInputIndex::WEIGHT_0;
    return static_cast<size_t>(first) + matrix * MOE_INPUTS_PER_MATRIX;
}

// A u3 weight matrix as the u3 kernels (moe_3gemm_int3.cl, moe_3gemm_int3_gemm.cl) take it
struct Int3MatrixDesc {
    size_t k;
    size_t n;
    size_t weight_group_size;  // 0 if the weights, scales or zero point don't suit the u3 kernels
    bool zp_scalar;
    ov::element::Type zp_type;
};

// matrix: 0 gate, 1 up, 2 down. u3 weights with f16 scales and, if the MoE has one, a u8/i8 zero point that is shared
// by all the experts or has the scales' shape.
inline Int3MatrixDesc get_int3_matrix_desc(const RuntimeParams& params, size_t matrix, bool shared = false) {
    const auto& config = params.typed_desc<cldnn::moe_3gemm_fused_compressed>()->_config;
    const bool down = matrix == 2;
    const size_t k = down ? config.inter_size : config.hidden_size;
    const size_t n = down ? config.hidden_size : config.inter_size;
    const size_t experts = shared ? 1 : config.num_expert;
    const auto& weight = params.get_input_layout(moe_weight_input(matrix, shared));
    const auto& scale = params.get_input_layout(moe_weight_input(matrix, shared) + 1);
    Int3MatrixDesc desc{k, n, 0, false, ov::element::dynamic};
    if (weight.data_type != ov::element::u3 || scale.data_type != ov::element::f16) {
        return desc;
    }
    const size_t weight_groups = scale.count() / (experts * n);
    if (weight_groups == 0 || scale.count() != experts * n * weight_groups || k % weight_groups != 0) {
        return desc;
    }
    if (config.has_zp) {
        const auto& zp = params.get_input_layout(moe_weight_input(matrix, shared) + 2);
        desc.zp_scalar = zp.count() == 1;
        desc.zp_type = zp.data_type;
        if (!cldnn::one_of(zp.data_type, {ov::element::u8, ov::element::i8}) || (!desc.zp_scalar && zp.count() != scale.count())) {
            return desc;
        }
    }
    desc.weight_group_size = k / weight_groups;
    return desc;
}

// Whether the u3 kernel configuration of a projection, taken from reference matrix ref, fits matrix desc too
inline bool same_int3_layout(const Int3MatrixDesc& desc, const Int3MatrixDesc& ref) {
    return desc.weight_group_size == ref.weight_group_size && desc.zp_scalar == ref.zp_scalar && desc.zp_type == ref.zp_type;
}

}  // namespace ov::intel_gpu::ocl
