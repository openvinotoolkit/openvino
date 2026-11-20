// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::ocl {

// Front end of the MoE expert GEMMs on int8 activations, for every weight type (moe_3gemm_dyn_quan.cl). Same scheme as
// dynamic_quantize_gpu_opt: symmetric int8 per group with f16 scales [rows, groups].
class MoE3GemmDynQuan {
public:
    enum class Projection { GATE_UP, DOWN };

    static constexpr size_t SIMD = 16;

    // Whether dynamic quantization is enabled for this MoE and the kernels support it: f16 hidden states and an
    // activation group (see get_activation_group_size()) for both projections.
    static bool is_supported(const RuntimeParams& params);

    // Input length of the projection: hidden_size for gate/up, inter_size for down
    static size_t get_k(const cldnn::MOECompressed::Config& config, Projection proj) {
        return proj == Projection::DOWN ? config.inter_size : config.hidden_size;
    }

    // Values per quantization group of the projection's input, 0 when it is not quantized: the largest of 128 and 64
    // that is at most the configured dynamic quantization group and divides the weight group.
    static size_t get_activation_group_size(const RuntimeParams& params, Projection proj);
};

// int8 quantization of the input of a projection: the hidden states for gate/up, act(gate) * up for down
class MoE3GemmDynQuanQuantize : public KernelGenerator {
public:
    explicit MoE3GemmDynQuanQuantize(MoE3GemmDynQuan::Projection proj)
        : KernelGenerator("moe_3gemm_dyn_quan", proj == MoE3GemmDynQuan::Projection::DOWN ? "quantize_down" : "quantize_gate_up"),
          m_proj(proj) {}

protected:
    MoE3GemmDynQuan::Projection m_proj;

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override;

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& params) const override {
        return {};
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt_params) {}};
    }
};

}  // namespace ov::intel_gpu::ocl
