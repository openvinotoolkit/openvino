// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_int3_gemm.hpp"

#include <algorithm>

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"
#include "moe_3gemm_int3_weights.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

MoE3GemmInt3Gemm::Shape MoE3GemmInt3Gemm::get_shape(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    Shape shape;
    static_cast<MoE3GemmDynQuan::Shape&>(shape) = MoE3GemmDynQuan::get_shape(params, proj);
    shape.n = proj == Projection::DOWN ? config.hidden_size : config.inter_size;
    shape.weight_group_size = get_group_size(config, shape.k);
    return shape;
}

// Requirements shared by all moe_3gemm_int3_gemm kernels. They need SIMD16 DPAS and the 2D block reads of Xe2+, read
// f16 scales, and take the zero point as u8/i8 in the u3 range [0, 7], subtracted from the weights while they are
// decoded. The activations are quantized to int8, which is accurate enough for
// zero-centered weights only, so the weights need a zero point. The tiled kernels consume an activation group as pairs
// of 32-value chunks, which the 64- and 128-value groups of MoE3GemmDynQuan are. Weight groups of a multiple of 64
// values make hidden_size and inter_size, so K and N of both projections, multiples of 64 too.
bool MoE3GemmInt3Gemm::is_supported(const RuntimeParams& params) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto& info = params.get_device_info();
    if (info.arch < gpu_arch::xe2 || !info.supports_immad || !config.has_zp || !MoE3GemmDynQuan::is_supported(params)) {
        return false;
    }
    // Gate and up share one kernel configuration, so up has to match gate
    for (size_t matrix = 0; matrix < 3; ++matrix) {
        const auto desc = get_int3_matrix_desc(params, matrix);
        if (desc.weight_group_size == 0 || !same_int3_layout(desc, get_int3_matrix_desc(params, matrix == 1 ? 0 : matrix))) {
            return false;
        }
    }
    for (auto proj : {Projection::GATE_UP, Projection::DOWN}) {
        const auto shape = get_shape(params, proj);
        if (shape.weight_group_size % 64 != 0 || shape.group_size % 64 != 0) {
            return false;
        }
    }
    return true;
}

size_t MoE3GemmInt3Gemm::get_max_tiles(size_t num_experts, size_t rows) {
    return std::min(num_experts, rows) + rows / TILE_ROWS;
}

JitConstants MoE3GemmInt3Gemm::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto shape = get_shape(params, m_proj);
    const auto weights = get_int3_matrix_desc(params, m_proj == Projection::DOWN ? 2 : 0);
    jit.make("K_SIZE", shape.k);
    jit.make("N_SIZE", shape.n);
    jit.make("GROUP_SIZE", shape.group_size);
    jit.make("WEI_GROUP_SIZE", shape.weight_group_size);
    jit.make("ZP_SCALAR", weights.zp_scalar ? 1 : 0);
    jit.make("ZP_TYPE", weights.zp_type == ov::element::i8 ? "char" : "uchar");
    jit.make("SG_M", TILE_SG_M);
    jit.make("TILE_M", TILE_ROWS / TILE_SG_M);
    jit.make("NB", TILE_NB);
    if (m_gate) {
        jit.make("GATE_ACT_MUL", 1);
        add_gate_activation_jit(jit, params.typed_desc<moe_3gemm_fused_compressed>()->_config.activation_type);
    }
    return jit;
}

// The tiles of TILE_M rows x NB column blocks need 256 registers per thread
std::string MoE3GemmInt3Gemm::get_build_options(const RuntimeParams& params) const {
    return KernelGenerator::get_build_options(params) + " -cl-intel-256-GRF-per-thread";
}

}  // namespace ov::intel_gpu::ocl
