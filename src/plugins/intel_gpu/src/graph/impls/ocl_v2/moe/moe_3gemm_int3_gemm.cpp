// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_int3_gemm.hpp"

#include <algorithm>

#include "common_utils/device_queries.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"
#include "moe_3gemm_int3_weights.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

// int8 activations are accurate enough for zero-centered weights only, so the weights need a zero point. Weight and
// activation groups of a multiple of 64 values make K and N of both projections multiples of 64 as well.
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
    const auto groups_of_64 = [&](Projection proj) {
        return get_group_size(config, MoE3GemmDynQuan::get_k(config, proj)) % 64 == 0 && MoE3GemmDynQuan::get_activation_group_size(params, proj) % 64 == 0;
    };
    return groups_of_64(Projection::GATE_UP) && groups_of_64(Projection::DOWN);
}

// On Xe2, gate and up run faster on one column block in the 128 GRF mode (twice the threads per EU) than on two
// blocks in 256 GRF; down is faster on two. The TILE_SG_M subgroups split the 32-value chunks of an activation group
// times the blocks among them, so one block takes groups of 128 values. See get_build_options().
size_t MoE3GemmInt3Gemm::get_tile_nb(const RuntimeParams& params, Projection proj) {
    const bool one_block = params.get_device_info().arch < gpu_arch::xe3 && proj == Projection::GATE_UP;
    return one_block && MoE3GemmDynQuan::get_activation_group_size(params, proj) / 32 % TILE_SG_M == 0 ? 1 : 2;
}

size_t MoE3GemmInt3Gemm::get_max_tiles(size_t num_experts, size_t rows) {
    return std::min(num_experts, rows) + rows / TILE_ROWS;
}

JitConstants MoE3GemmInt3Gemm::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const bool down = m_proj == Projection::DOWN;
    const size_t k = MoE3GemmDynQuan::get_k(config, m_proj);
    const auto weights = get_int3_matrix_desc(params, down ? 2 : 0);
    jit.make("K_SIZE", k);
    jit.make("N_SIZE", down ? config.hidden_size : config.inter_size);
    jit.make("GROUP_SIZE", MoE3GemmDynQuan::get_activation_group_size(params, m_proj));
    jit.make("WEI_GROUP_SIZE", get_group_size(config, k));
    jit.make("ZP_SCALAR", weights.zp_scalar ? 1 : 0);
    jit.make("ZP_TYPE", weights.zp_type == ov::element::i8 ? "char" : "uchar");
    jit.make("SG_M", TILE_SG_M);
    jit.make("TILE_M", TILE_ROWS / TILE_SG_M);
    jit.make("NB", get_tile_nb(params, m_proj));
    if (m_gate) {
        jit.make("GATE_ACT_MUL", 1);
        jit.make("MOE_GATE_ACT_NATIVE", 1);
        add_gate_activation_jit(jit, config.activation_type);
    }
    return jit;
}

// On Xe2 two column blocks need 256 GRFs and one fits the default 128. On Xe3 the compiler's choice suits gate and
// up, but down runs faster with 192 than with the 128 it picks.
std::string MoE3GemmInt3Gemm::get_build_options(const RuntimeParams& params) const {
    auto options = KernelGenerator::get_build_options(params);
    if (params.get_device_info().arch < gpu_arch::xe3) {
        return get_tile_nb(params, m_proj) == 2 ? options + " -cl-intel-256-GRF-per-thread" : options;
    }
    if (m_proj != Projection::DOWN) {
        return options;
    }
    const auto& program = params.get_program();
    if (query_register_file_size_option_supported(program.get_engine(), program.get_config())) {
        options += " -ze-exp-register-file-size 192";
    }
    return options;
}

}  // namespace ov::intel_gpu::ocl
