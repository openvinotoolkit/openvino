// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Batched GEMV decode of u3 experts (moe_3gemm_int3.cl)
class MoE3GemmInt3Gemv : public KernelGenerator {
public:
    enum class Projection { GATE_UP, DOWN };

    // Work-groups of SGN subgroups of SIMD lanes, each subgroup computing get_nb() output columns of one
    // (token, top-k slot) pair.
    static constexpr size_t SGN = 8;
    static constexpr size_t SIMD = 16;

    explicit MoE3GemmInt3Gemv(Projection proj) : KernelGenerator("moe_3gemm_int3", proj == Projection::DOWN ? "gemv_down" : "gemv_gate_up"), m_proj(proj) {}

    // Whether the kernels support this MoE: u3 weights, f16 activations and scales, a u8/i8 zero point that is scalar
    // or shaped like the scales (or none), K a multiple of 512 and weight groups a multiple of 32 values. A shared
    // expert needs u3 weights of the same layout.
    static bool is_supported(const RuntimeParams& params);

    // Output columns per subgroup of a projection, tied to the GPU size
    static size_t get_nb(const RuntimeParams& params, Projection proj);

    // Most tokens per step that decode through the u3 batched GEMV unless GPU_MOE_BATCHED_GEMV_THRESHOLD is set
    static size_t get_default_threshold(const RuntimeParams& params);

protected:
    Projection m_proj;

    // Whether the shared expert is fused into the kernels, as an extra slot per token
    static bool has_shared_expert(const RuntimeParams& params);

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

}  // namespace ov::intel_gpu::ocl
