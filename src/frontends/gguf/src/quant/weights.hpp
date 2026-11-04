// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "gguf.hpp"
#include "openvino/core/node_output.hpp"

namespace ov {
class Node;
class Model;
}  // namespace ov

namespace ov::frontend::gguf {

void normalize_u2_zero_points(const std::shared_ptr<ov::Model>& model);

// Rows of a ternary (Q2_0 family) decompressed weight gathered from its packed bytes before
// unpacking; nullptr when `data` is not such a weight.
std::shared_ptr<ov::Node> gather_packed_u2_rows(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& indices);
ov::Output<ov::Node> gather_weight_rows(const ov::Output<ov::Node>& data, const ov::Output<ov::Node>& indices);

// Element type of the zero-point constant for an asymmetric quantized weight.
// Q2_0 always uses u8. The raw-byte Q8_0_C requantization path separately uses temporary f16
// zero-points for its host dequantizer, including Q2_0 embedding/output tensors.
// Q4_K and Q4_1 matmul weights are faithfully decoded and requantized to u4 with an integer
// zero-point, which keeps the compressed-FullyConnected path available on CPU and GPU. Native
// Q5_K matmul weights use symmetric i8. token_embd/output keep an f16 zero-point.
ov::element::Type gguf_zero_point_type(const std::string& name, GgufTensorType qtype);

bool native_q5_k_is_symmetric(const std::string& name);

// A lossy weight approximation the frontend deliberately makes, reported to the user once so a
// later accuracy investigation starts from the right place.
enum class LossyWeightApproximation {
    // token_embd / output / Q6_K / Q5_K tensors requantized channel-wise to Q8_0_C.
    Q8_0_C_REQUANT,
    // Q4_K / Q4_1 asymmetric weights faithfully decoded and requantized group-wise to OpenVINO u4.
    Q4_REQUANT,
    // Q5_K asymmetric weights faithfully decoded and requantized group-wise to OpenVINO u8.
    Q5_K_REQUANT,
    Q5_K_SYMMETRIC_REQUANT,
};

// Warn -- ONCE per process and per approximation kind -- that weights are being converted with a
// lossy approximation, so generated text may differ slightly from the original GGUF weights.
//
// Once per process rather than once per model: the message describes a fixed property of the
// conversion strategy, not of one tensor or one file, and a model has thousands of affected
// weights (every Q4_K matmul), so anything finer-grained would bury the log. Call it at the point
// the approximation is actually PERFORMED, not where its parameters are queried.
void notify_lossy_weight_approximation(LossyWeightApproximation kind);

// The tensors extracted from one GGUF weight. `scales` and `zero_point` are empty for formats
// that do not use them.
struct WeightTensors {
    ov::Tensor weight;
    ov::Tensor scales;
    ov::Tensor zero_point;
};

bool same_weight_conversion_policy(const std::string& lhs, const std::string& rhs);

// Build the OpenVINO node for one extracted GGUF weight. Quantized weights become a
// low-bitness compressed subgraph (u4/u8 weights + zero-point + f16 scale, Convert ->
// Subtract -> Multiply -> Reshape), matching what the cgraph path produces; F16/F32 weights
// become a plain Constant. The returned node's output is f32 and feeds the translators.
std::shared_ptr<ov::Node> make_weight_node(const WeightTensors& tensors,
                                           GgufTensorType qtype,
                                           const std::string& name = "weight");

std::shared_ptr<ov::Node> make_weight_node(const GgufWeight& source);
// u8 shape {1, E, M, K/32, 17} of a packed MXFP4 expert tensor with ggml shape [..., E, M, K].
ov::Shape mxfp4_packed_shape(const ov::Shape& shape);

// Build the OpenVINO weight node directly from the raw GGUF weight bytes, as provided by a
// decoder (e.g. wrapping llama.cpp's tensor->data). `data` holds the bytes exactly as ggml
// laid them out; `qtype` is the ggml tensor type; `logical_shape` is the {rows, cols} element shape. This is the
// frontend entry point used by translate_weight: it runs the appropriate fill function to extract weights/scales/zp and
// builds the decompression subgraph. Throws on an unsupported quant type. The returned node's output is f32.
//
// `name` is the gguf tensor name (e.g. "token_embd.weight", "blk.0.ffn_down.weight"). It is
// used to decide channel-wise requantization to Q8_0_C for the embedding / output / Q6_K /
// Q5_K tensors, matching the llama.cpp ggml-openvino backend's CPU/GPU weight pipeline.
std::shared_ptr<ov::Node> make_weight_node(const ov::Tensor& data,
                                           GgufTensorType qtype,
                                           const ov::Shape& logical_shape,
                                           const std::string& name = "");
// Same, with `quant_type` given as the ggml type name ("F16", "Q4_0", "q6_K", ...).
std::shared_ptr<ov::Node> make_weight_node(const ov::Tensor& data,
                                           const std::string& quant_type,
                                           const ov::Shape& logical_shape,
                                           const std::string& name = "");

// Map a ggml quant type name (e.g. "Q4_K") to its GgufTensorType id. Throws if unknown.
GgufTensorType gguf_type_from_name(const std::string& quant_type);

}  // namespace ov::frontend::gguf
