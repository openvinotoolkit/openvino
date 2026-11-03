// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Batched GEMV decode of u3 experts (moe_3gemm_int3.cl)
enum class Int3Projection { GATE_UP, DOWN };

// Work-groups of INT3_GEMV_SGN subgroups of INT3_GEMV_SIMD lanes, each subgroup computing int3_gemv_nb() output
// columns of one (token, top-k slot) pair.
constexpr size_t INT3_GEMV_SGN = 8;
constexpr size_t INT3_GEMV_SIMD = 16;

// Whether the u3 batched GEMV kernels support this MoE: f16 activations and scales, a u8/i8 zero point that is scalar or
// shaped like the scales (or none), K a multiple of 512 and weight groups a multiple of 32 values. A shared expert
// needs u3 weights of the same layout.
bool int3_gemv_supported(const RuntimeParams& params);

// Whether the shared expert is fused into the kernels, as an extra slot per token
bool int3_gemv_has_shared_expert(const RuntimeParams& params);

// Output columns per subgroup of a projection, tied to the GPU size
size_t int3_gemv_nb(const RuntimeParams& params, Int3Projection proj);

// Most tokens per step that decode through the u3 batched GEMV unless GPU_MOE_BATCHED_GEMV_THRESHOLD is set
size_t int3_gemv_default_threshold(const RuntimeParams& params);

class MoE3GemmInt3Gemv : public KernelGenerator {
public:
    explicit MoE3GemmInt3Gemv(Int3Projection proj)
        : KernelGenerator("moe_3gemm_int3", proj == Int3Projection::DOWN ? "gemv_down" : "gemv_gate_up"),
          m_proj(proj) {}

protected:
    Int3Projection m_proj;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

}  // namespace ov::intel_gpu::ocl
