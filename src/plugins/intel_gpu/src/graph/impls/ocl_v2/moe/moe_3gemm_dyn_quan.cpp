// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_dyn_quan.hpp"

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "moe_3gemm_base.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

MoE3GemmDynQuan::Shape MoE3GemmDynQuan::get_shape(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    Shape shape;
    shape.k = proj == Projection::DOWN ? config.inter_size : config.hidden_size;
    const auto& exec_config = params.get_program().get_config();
    const uint64_t dyn_quan_group_size = exec_config.get_dynamic_quantization_group_size();
    const size_t weight_group_size = get_group_size(config, shape.k);
    if (dyn_quan_group_size == 0 || exec_config.get_dynamic_quantization_group_size_max() < dyn_quan_group_size) {
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
    const auto& id = params.desc->id;
    if (params.get_input_layout(static_cast<size_t>(MOE3GemmInputIndex::HIDDEN_STATES)).data_type != ov::element::f16) {
        GPU_DEBUG_TAGGED(TRACE, "dynquan") << id << " turned off: only f16 activations are supported" << std::endl;
        return false;
    }
    const auto& exec_config = params.get_program().get_config();
    const uint64_t dyn_quan_group_size = exec_config.get_dynamic_quantization_group_size();
    if (dyn_quan_group_size == 0) {
        GPU_DEBUG_TAGGED(TRACE, "dynquan") << id << " turned off: dynamic_quantization_group_size is 0" << std::endl;
        return false;
    }
    if (exec_config.get_dynamic_quantization_group_size_max() < dyn_quan_group_size) {
        GPU_DEBUG_TAGGED(TRACE, "dynquan") << id << " turned off: group_size " << dyn_quan_group_size << " is larger than the max "
                                           << exec_config.get_dynamic_quantization_group_size_max() << std::endl;
        return false;
    }
    for (auto proj : {Projection::GATE_UP, Projection::DOWN}) {
        const auto shape = get_shape(params, proj);
        if (shape.group_size == 0) {
            const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
            GPU_DEBUG_TAGGED(TRACE, "dynquan") << id << " turned off: no activation group of 128 or 64 values that is at most group_size "
                                               << dyn_quan_group_size << " divides the " << (proj == Projection::DOWN ? "down" : "gate/up")
                                               << " weight group of " << get_group_size(config, shape.k) << std::endl;
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
