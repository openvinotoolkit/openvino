// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "../utils/kernel_generator.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl {

// Dynamic quantization of the activations of the MoE expert GEMMs (moe_3gemm_dyn_quan.cl), the front end of the expert
// GEMMs that run on int8 activations, whatever their weight type: the hidden states are quantized once per token and
// gathered into their rows sorted by expert for gate and up, and act(gate) * up is quantized again for down. The
// quantization is that of the FullyConnected layers (dynamic_quantize_gpu_opt): symmetric int8 per group, with an f16
// dequantization scale and the i32 sum of the quantized values of each group, both [rows, groups].
class MoE3GemmDynQuan {
public:
    enum class Projection { GATE_UP, DOWN };

    struct Shape {
        size_t k = 0;           // input length of the projection
        size_t group_size = 0;  // values per quantization group, 0 when the input is not quantized
        size_t groups() const {
            return k / group_size;
        }
    };

    static constexpr size_t SIMD = 16;

    // Whether dynamic quantization is enabled for this MoE and the kernels support it: f16 hidden states and an
    // activation group (see get_shape()) for both projections.
    static bool is_supported(const RuntimeParams& params);

    // The activation group is the largest of 128 and 64 values that is no larger than the configured dynamic
    // quantization group and divides the weight group of the projection: 128 for the per-token default of Xe2+, as
    // for the u8/i8 FullyConnected layers.
    static Shape get_shape(const RuntimeParams& params, Projection proj);

    // Values per weight scale group of the projection, 0 when the scales don't split K into equal groups
    static size_t get_weight_group_size(const RuntimeParams& params, Projection proj);

    // Work-group size of the gather: one work-item per 16 bytes of a row and per group
    static size_t get_gather_items(const Shape& gate_up);
};

// int8 quantization of the gate/up input (the hidden states), or with swiglu of act(gate) * up for the down projection
class MoE3GemmDynQuanQuantize : public KernelGenerator {
public:
    explicit MoE3GemmDynQuanQuantize(bool swiglu) : KernelGenerator("moe_3gemm_dyn_quan", swiglu ? "quantize_swiglu" : "quantize"), m_swiglu(swiglu) {}

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

// Copies the int8 hidden states and the scales and sums of each token into its rows sorted by expert
class MoE3GemmDynQuanGather : public KernelGenerator {
public:
    MoE3GemmDynQuanGather() : KernelGenerator("moe_3gemm_dyn_quan", "gather") {}

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
