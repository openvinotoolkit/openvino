// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "moe_3gemm_dyn_quan.hpp"

namespace ov::intel_gpu::ocl {

// Expert GEMMs of the grouped GEMM path for u3 experts (moe_3gemm_int3_gemm.cl), on the int8 activations of the
// MoE3GemmDynQuan front end. The gate GEMM computes act(gate) * up in its epilogue, from the output of the up GEMM.
class MoE3GemmInt3Gemm : public KernelGenerator {
public:
    using Projection = MoE3GemmDynQuan::Projection;

    struct Shape : MoE3GemmDynQuan::Shape {
        size_t n = 0;
        size_t weight_group_size = 0;  // a multiple of the activation group
    };

    static constexpr size_t SIMD = 16;
    // Rows per entry of the host-built tile list of the expert mode kernels
    static constexpr size_t TILE_ROWS = 64;
    // Expert mode work-groups: TILE_SG_M subgroups of TILE_ROWS / TILE_SG_M rows x TILE_NB 16-column blocks. N is a
    // multiple of 64 and a quantization group of 64 or more values, so they fit every supported shape.
    static constexpr size_t TILE_SG_M = 4;
    static constexpr size_t TILE_NB = 2;
    // Experts with at most this many rows go through the per-row kernels
    static constexpr int ROW_MODE_MAX_ROWS = 4;

    // gate: the gate GEMM of GATE_UP, with the act(gate) * up epilogue, rather than the up GEMM
    MoE3GemmInt3Gemm(Projection proj, bool row_mode, bool gate = false)
        : KernelGenerator("moe_3gemm_int3_gemm",
                          std::string(proj == Projection::DOWN ? "down" : (gate ? "gate" : "up")) + (row_mode ? "_row" : "_tiled")),
          m_proj(proj),
          m_row_mode(row_mode),
          m_gate(gate) {}

    // Whether the kernels support this MoE: activations quantized by MoE3GemmDynQuan, u3 weights with a u8/i8 zero
    // point (scalar or shaped like the scales), f16 scales and weight groups a multiple of 64 values, on Xe2+ with DPAS.
    static bool is_supported(const RuntimeParams& params);

    static Shape get_shape(const RuntimeParams& params, Projection proj);

    // Subgroups splitting K in the per-row kernels
    static size_t get_row_sg_k(const Shape& shape);
    // Upper bound of the tile list length: every expert with rows has at most one partial tile
    static size_t get_max_tiles(size_t num_experts, size_t rows);

protected:
    Projection m_proj;
    bool m_row_mode;
    bool m_gate;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

}  // namespace ov::intel_gpu::ocl
