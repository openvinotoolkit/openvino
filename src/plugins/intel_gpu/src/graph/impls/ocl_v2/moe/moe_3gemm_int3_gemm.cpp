// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_3gemm_int3_gemm.hpp"

#include <algorithm>

#include "common_utils/jitter.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"
#include "moe_3gemm_base.hpp"
#include "ocl_v2/utils/jitter.hpp"

using namespace cldnn;  // TODO: Remove once namespaces are aligned

namespace ov::intel_gpu::ocl {

namespace {

// Weight matrices in input order (gate, up, down), each followed by its scale and zero point
constexpr size_t INPUTS_PER_MATRIX = 3;

size_t weight_input(size_t matrix) {
    return static_cast<size_t>(MOE3GemmInputIndex::WEIGHT_0) + matrix * INPUTS_PER_MATRIX;
}

constexpr size_t ROW_SG_K = 4;
constexpr size_t TILE_SG_M = 4;

}  // namespace

MoE3GemmInt3Gemm::Shape MoE3GemmInt3Gemm::get_shape(const RuntimeParams& params, Projection proj) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    Shape shape;
    static_cast<MoE3GemmDynQuan::Shape&>(shape) = MoE3GemmDynQuan::get_shape(params, proj);
    shape.n = proj == Projection::DOWN ? config.hidden_size : config.inter_size;
    shape.weight_group_size = MoE3GemmDynQuan::get_weight_group_size(params, proj);
    return shape;
}

// Requirements shared by all moe_3gemm_int3_gemm kernels. They need SIMD16 DPAS and the 2D block reads of Xe2+, read
// f16 scales, and take the zero point as u8/i8. The activations are quantized to int8, which is accurate enough for
// zero-centered weights only, so the weights need a zero point. The tiled kernels consume an activation group as pairs
// of 32-value chunks, which the 64- and 128-value groups of MoE3GemmDynQuan are.
bool MoE3GemmInt3Gemm::is_supported(const RuntimeParams& params) {
    const auto& config = params.typed_desc<moe_3gemm_fused_compressed>()->_config;
    const auto& info = params.get_device_info();
    if (info.arch < gpu_arch::xe2 || !info.supports_immad || !config.has_zp || !MoE3GemmDynQuan::is_supported(params)) {
        return false;
    }
    // Gate and up share one kernel configuration, so up has to match gate
    for (size_t matrix = 0; matrix < 3; ++matrix) {
        const size_t ref = matrix == 1 ? 0 : matrix;
        const auto& weight = params.get_input_layout(weight_input(matrix));
        const auto& scale = params.get_input_layout(weight_input(matrix) + 1);
        const auto& zp = params.get_input_layout(weight_input(matrix) + 2);
        const auto& ref_scale = params.get_input_layout(weight_input(ref) + 1);
        const auto& ref_zp = params.get_input_layout(weight_input(ref) + 2);
        if (weight.data_type != ov::element::u3 || scale.data_type != ov::element::f16 || scale.count() != ref_scale.count() ||
            !one_of(zp.data_type, {ov::element::u8, ov::element::i8}) || zp.data_type != ref_zp.data_type || zp.count() != ref_zp.count() ||
            (zp.count() != 1 && zp.count() != scale.count())) {
            return false;
        }
    }
    for (auto proj : {Projection::GATE_UP, Projection::DOWN}) {
        const auto shape = get_shape(params, proj);
        if (shape.weight_group_size % 64 != 0 || shape.group_size % 64 != 0 || shape.n % 16 != 0) {
            return false;
        }
    }
    return true;
}

size_t MoE3GemmInt3Gemm::get_row_sg_k(const Shape& shape) {
    return std::min(ROW_SG_K, shape.groups());
}

size_t MoE3GemmInt3Gemm::get_tile_nb(const Shape& shape) {
    return shape.n % 32 == 0 ? 2 : 1;
}

// The subgroups split the 32-value granules of a quantization group (for all column blocks) between them
size_t MoE3GemmInt3Gemm::get_tile_sg_m(const Shape& shape) {
    return std::min(TILE_SG_M, get_tile_nb(shape) * shape.group_size / 32);
}

size_t MoE3GemmInt3Gemm::get_max_tiles(size_t num_experts, size_t rows) {
    return std::min(num_experts, rows) + rows / TILE_ROWS;
}

JitConstants MoE3GemmInt3Gemm::get_jit_constants(const RuntimeParams& params) const {
    auto jit = KernelGenerator::get_jit_constants(params);
    const auto shape = get_shape(params, m_proj);
    const auto& zp = params.get_input_layout(weight_input(m_proj == Projection::DOWN ? 2 : 0) + 2);
    jit.make("ROW_MODE", m_row_mode ? 1 : 0);
    jit.make("K_SIZE", shape.k);
    jit.make("N_SIZE", shape.n);
    jit.make("GROUP_SIZE", shape.group_size);
    jit.make("WEI_GROUP_SIZE", shape.weight_group_size);
    jit.make("HAS_ZP", 1);
    jit.make("ZP_SCALAR", zp.count() == 1 ? 1 : 0);
    jit.make("ZP_TYPE", zp.data_type == ov::element::i8 ? "char" : "uchar");
    jit.make("SG_K", get_row_sg_k(shape));
    jit.make("SG_M", get_tile_sg_m(shape));
    jit.make("TILE_M", TILE_ROWS / get_tile_sg_m(shape));
    jit.make("NB", get_tile_nb(shape));
    return jit;
}

// The expert mode tiles of TILE_M rows x NB column blocks need 256 registers per thread
std::string MoE3GemmInt3Gemm::get_build_options(const RuntimeParams& params) const {
    auto options = KernelGenerator::get_build_options(params);
    if (!m_row_mode) {
        options += " -cl-intel-256-GRF-per-thread";
    }
    return options;
}

}  // namespace ov::intel_gpu::ocl
