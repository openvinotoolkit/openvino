// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// vk_ir: plugin-agnostic graph IR consumed by the Vulkan runtime core.
//
// The Vulkan runtime (vk_program, vk_network, vk_engine, ...) is the
// standalone core: it knows buffers, kernels and push constants, and nothing
// about openvino core graph objects. ov::Model graphs reach it through this
// IR, lowered on the plugin side by VkModelConverter
// (src/plugin/vk_model_converter.{hpp,cpp}).

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace ov::core {
namespace vulkan {
namespace cross_platform {

enum class ir_op {
    parameter,  // model input, allocated as host-visible buffer
    constant,   // f32 data baked into const memory at build time
    result,     // model output, no kernel
    relu,
    add,
    max_pool,
    avg_pool,
    convolution,
    matmul,
    mul,
    sub,
    div,
    sigmoid,
    tanh,
    leaky_relu,
    transpose,
    concat,
    softmax,
    reshape,      // no kernel: flat f32 buffers are reinterpreted in place
    reduce_mean,
    reduce_sum,
    reduce_max,
    gelu,         // tanh approximation: 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))
    swiglu,       // silu(a)*b, silu(x) = x*sigmoid(x); two same-shape inputs
    quick_gelu,   // x * sigmoid(1.702x)
    rms_norm,     // out = x/sqrt(mean(x^2,axis)+eps) * weight; alpha holds eps
    pad,          // constant fill; pads_begin/pads_end per dim, alpha = fill
    crop,         // per-dim window; begin offsets in pads_begin
    causal_softmax,  // softmax over the last axis with j>i masked to -inf
    rope,         // rotary embedding: x[...,D] x cos/sin[L,D/2]; halves convention
    cache_write,  // cache[B,S,D] rows [pos, pos+L) := new[B,L,D]; pos = alpha
    argmax,       // index (f32) of the max along the last axis
};

struct ir_pool_params {
    std::vector<size_t> kernel;
    std::vector<size_t> strides;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;  // Pad op: per-dim end padding
};

// Quantized weight constant: the raw on-disk GGUF block payload plus the
// quant type id (see gguf_tensor_type in the reader). No dequant happens on
// the host; the consuming kernel unpacks the blocks natively in-shader.
struct ir_quant_const {
    uint32_t type = 0;          // gguf quant type id (e.g. 2 = Q4_0)
    std::vector<uint8_t> bytes; // raw block bytes, exactly as stored in the file
};

struct ir_node {
    std::string id;                    // unique id, doubles as output buffer id
    ir_op op = ir_op::parameter;
    std::vector<std::string> inputs;   // producer buffer ids (canonical)
    ir_pool_params pool;
    bool matmul_transpose_b = false;   // MatMul: second input is the transposed weight matrix [N,K]
    float alpha = 0.0f;                // op scalar attribute: leaky_relu slope, rms_norm eps, pad fill value
    std::vector<size_t> transpose_order;  // Transpose: permutation of the input axes
    uint32_t axis = 0;                 // Concat / Softmax / Reduce working axis
};

struct ir_graph {
    // Topologically sorted nodes, sources first.
    std::vector<ir_node> nodes;
    // Buffer id -> f32 tensor shape (params, constants and op outputs).
    std::map<std::string, std::vector<size_t>> tensor_shapes;
    // Constant buffer id -> f32 payload.
    std::map<std::string, std::vector<float>> constant_data;
    // Quantized constant buffer id -> raw block payload (mutually exclusive
    // with constant_data for the same id).
    std::map<std::string, ir_quant_const> quant_constants;
    // Model input/output buffer ids in port order.
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

}  // namespace cross_platform
}  // namespace vulkan
}  // namespace ov