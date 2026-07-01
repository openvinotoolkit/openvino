// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "gguf.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Build the OpenVINO node for a GGUF weight with base name `base` (the tensor name without
// the trailing ".weight", e.g. "blk.0.attn_q" or "token_embd"). Quantized weights become a
// low-bitness compressed subgraph (u4/u8 weights + zero-point + f16 scale, Convert ->
// Subtract -> Multiply -> Reshape), matching what the cgraph path produces; F16/F32 weights
// become a plain Constant. The returned node's output is f32 and feeds the translators.
//
// `weights` holds the parser output (tensors by gguf name; quantized tensors expanded to
// "<base>.weight" + "<base>.scales" + "<base>.biases"). `qtypes` maps "<base>.qtype" ->
// gguf_tensor_type.
std::shared_ptr<ov::Node> make_weight_node(const std::string& base,
                                           const std::unordered_map<std::string, ov::Tensor>& weights,
                                           const std::unordered_map<std::string, gguf_tensor_type>& qtypes);

// Build the OpenVINO weight node directly from the raw GGUF weight bytes, as provided by a
// decoder (e.g. wrapping llama.cpp's tensor->data). `data` holds the bytes exactly as ggml
// laid them out; `quant_type` is the ggml type name ("F16", "F32", "BF16", "Q4_0", "Q4_K",
// "Q6_K", ...); `logical_shape` is the {rows, cols} element shape. This is the frontend
// entry point used by translate_weight: it runs the appropriate fill function to extract
// weights/scales/zp and builds the decompression subgraph. Throws on an unsupported quant
// type. The returned node's output is f32.
//
// `name` is the gguf tensor name (e.g. "token_embd.weight", "blk.0.ffn_down.weight"). It is
// used to decide channel-wise requantization to Q8_0_C for the embedding / output / Q6_K /
// Q5_K tensors, matching the llama.cpp ggml-openvino backend's CPU/GPU weight pipeline.
std::shared_ptr<ov::Node> make_weight_node(const ov::Tensor& data,
                                           const std::string& quant_type,
                                           const ov::Shape& logical_shape,
                                           const std::string& name = "");

// Map a ggml quant type name (e.g. "Q4_K") to its gguf_tensor_type id. Throws if unknown.
gguf_tensor_type gguf_type_from_name(const std::string& quant_type);

// Split a fused attention weight `<base>` (rows = n_q + n_k + n_v output features, i.e.
// the GGUF attn_qkv tensor) into three decompressed weight nodes for q/k/v by slicing the
// row dimension (rows are block-independent in the quant layout). `base` is e.g.
// "blk.0.attn_qkv". n_q/n_k/n_v are the output-feature counts. Returns {q, k, v} nodes.
std::array<std::shared_ptr<ov::Node>, 3> make_fused_qkv_weights(
    const std::string& base,
    const std::unordered_map<std::string, ov::Tensor>& weights,
    const std::unordered_map<std::string, gguf_tensor_type>& qtypes,
    size_t n_q,
    size_t n_k,
    size_t n_v);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
