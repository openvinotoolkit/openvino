// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_dyn_quan.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

size_t MoE3GemmDynQuan::get_weight_group_size(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const bool down = proj == Projection::DOWN;
    const size_t k = down ? config.inter_size : config.hidden_size;
    const size_t n = down ? config.hidden_size : config.inter_size;
    const auto scale_input = static_cast<size_t>(down ? MOE3GemmInputIndex::SCALE_2 : MOE3GemmInputIndex::SCALE_0);
    const size_t scale_count = params.get_input_layout(scale_input).count();
    const size_t weight_groups = scale_count / (config.num_expert * n);
    if (weight_groups == 0 || scale_count != config.num_expert * n * weight_groups || k % weight_groups != 0) {
        return 0;
    }
    return k / weight_groups;
}

MoE3GemmDynQuan::Shape MoE3GemmDynQuan::get_shape(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    Shape shape;
    shape.k = proj == Projection::DOWN ? config.inter_size : config.hidden_size;
    const auto& exec_config = params.get_program().get_config();
    const uint64_t dyn_quan_group_size = exec_config.get_dynamic_quantization_group_size();
    const size_t weight_group_size = get_weight_group_size(params, proj);
    if (dyn_quan_group_size == 0 || exec_config.get_dynamic_quantization_group_size_max() < dyn_quan_group_size || weight_group_size == 0) {
        return shape;
    }
    for (size_t candidate : {size_t{128}, size_t{64}}) {
        if (candidate <= dyn_quan_group_size && weight_group_size % candidate == 0) {
            shape.group_size = candidate;
            break;
        }
    }
    return shape;
}

bool MoE3GemmDynQuan::is_supported(const RuntimeParams& params) {
    if (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)).data_type != ov::element::f16) {
        return false;
    }
    for (auto proj : {Projection::GATE_UP, Projection::DOWN}) {
        if (get_shape(params, proj).group_size == 0) {
            return false;
        }
    }
    return true;
}

JitConstants MoE3GemmDynQuanQuantize::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto shape = MoE3GemmDynQuan::get_shape(params, m_proj);
    jit.make("GROUP_SIZE", shape.group_size);
    jit.make("VEC_SIZE", shape.group_size / MoE3GemmDynQuan::SIMD);
    return jit;
}

}  // namespace ov::intel_gpu::ocl
