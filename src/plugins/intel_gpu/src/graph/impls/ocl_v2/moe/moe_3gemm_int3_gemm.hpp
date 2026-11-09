// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Expert GEMMs of the grouped GEMM path for u3 experts (moe_3gemm_int3_gemm.cl), on dynamically quantized int8
// activations: the hidden states are quantized once per token and gathered per expert for gate and up, and
// act(gate) * up is quantized again for down.
class MoE3GemmInt3Gemm : public KernelGenerator {
public:
    enum class Projection { GATE_UP, DOWN };

    struct Shape {
        size_t k = 0;
        size_t n = 0;
        size_t weight_group_size = 0;
        size_t group_size = 0;  // activation quantization group, divides weight_group_size
    };

    static constexpr size_t SIMD = 16;
    // Rows per entry of the host-built tile list of the expert mode kernels
    static constexpr size_t TILE_ROWS = 64;
    // Experts with at most this many rows go through the per-row kernels
    static constexpr int ROW_MODE_MAX_ROWS = 4;

    MoE3GemmInt3Gemm(Projection proj, bool row_mode)
        : KernelGenerator("moe_3gemm_int3_gemm", std::string(proj == Projection::DOWN ? "down" : "gate_up") + (row_mode ? "_row" : "_tiled")),
          m_proj(proj),
          m_row_mode(row_mode) {}

    // Whether the kernels support this MoE and dynamic quantization is enabled for it: u3 weights with a u8/i8 zero
    // point (scalar or shaped like the scales), f16 activations and scales, weight groups a multiple of 64 values,
    // N a multiple of 16, on Xe2+ with DPAS.
    static bool is_supported(const RuntimeParams& params);

    static Shape get_shape(const RuntimeParams& params, Projection proj);

    // Subgroups splitting K in the per-row kernels
    static size_t get_row_sg_k(const Shape& shape);
    // 16-column blocks per subgroup in the expert mode kernels
    static size_t get_tile_nb(const Shape& shape);
    // Subgroups per work-group in the expert mode kernels, each computing TILE_ROWS / get_tile_sg_m() rows
    static size_t get_tile_sg_m(const Shape& shape);
    // Work-group size of the int8 token gather: one work-item per 16 bytes of a row and per quantization variable
    static size_t get_gather_items(const Shape& gate_up);
    // Upper bound of the tile list length: every expert with rows has at most one partial tile
    static size_t get_max_tiles(size_t num_experts, size_t rows);

protected:
    Projection m_proj;
    bool m_row_mode;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

// int8 quantization of the gate/up input (the hidden states), or with swiglu of act(gate) * up for the down projection
class MoE3GemmInt3Quantize : public KernelGenerator {
public:
    explicit MoE3GemmInt3Quantize(bool swiglu) : KernelGenerator("moe_3gemm_int3_gemm", swiglu ? "quantize_swiglu" : "quantize"), m_swiglu(swiglu) {}

protected:
    bool m_swiglu;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

// Copies the int8 hidden states and their quantization variables of each token into its rows sorted by expert
class MoE3GemmInt3GatherQuantized : public KernelGenerator {
public:
    MoE3GemmInt3GatherQuantized() : KernelGenerator("moe_3gemm_int3_gemm", "gather_quantized") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

}  // namespace ov::intel_gpu::ocl
