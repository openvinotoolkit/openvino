// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/moe_3gemm_fused_compressed.hpp"

namespace ov::intel_gpu::ocl {

// Dynamic quantization of the activations of the MoE expert GEMMs (moe_3gemm_dyn_quan.cl), the front end of the expert
// GEMMs that run on int8 activations, whatever their weight type: the hidden states are quantized once per token and
// then gathered into their rows sorted by expert for gate and up, and act(gate) * up is quantized for down. The
// quantization is that of the FullyConnected layers (dynamic_quantize_gpu_opt): symmetric int8 per group, with an f16
// dequantization scale of each group, [rows, groups].
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

    // Values per quantization group of the projection's input, 0 when it is not quantized. The activation group is
    // the largest of 128 and 64 values that is no larger than the configured dynamic quantization group and divides the
    // weight group of the projection: 128 for the per-token default of Xe2+, as for the u8/i8 FullyConnected layers.
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
