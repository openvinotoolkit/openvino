// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>

#include "gguf.hpp"

namespace ov {
class Node;
}

namespace ov::frontend::gguf {

// Element type of the zero-point constant for an asymmetric quantized weight. Both ingest
// paths must agree on this: it decides whether the CPU folds the dequant into the MatMul.
// Q4_K defaults to an integer (u8) zero-point (faster, lossy); set the environment variable
// OV_GGUF_Q4_K_ZP_F16=1 to use a faithful f16 zero-point instead (llama.cpp's test-backend-ops
// CI does this to meet its accuracy tolerance).
ov::element::Type gguf_zero_point_type(const std::string& name, GgufTensorType qtype);

// A lossy weight approximation the frontend deliberately makes, reported to the user once so a
// later accuracy investigation starts from the right place.
enum class LossyWeightApproximation {
    // token_embd / output / Q6_K / Q5_K tensors requantized channel-wise to Q8_0_C.
    Q8_0_C_REQUANT,
    // Q4_K asymmetric weights expressed with an INTEGER (u8) zero-point, which forces each
    // sub-block's min to a multiple of its scale.
    INTEGER_ZERO_POINT,
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

// Build the OpenVINO node for one extracted GGUF weight. Quantized weights become a
// low-bitness compressed subgraph (u4/u8 weights + zero-point + f16 scale, Convert ->
// Subtract -> Multiply -> Reshape), matching what the cgraph path produces; F16/F32 weights
// become a plain Constant. The returned node's output is f32 and feeds the translators.
std::shared_ptr<ov::Node> make_weight_node(const WeightTensors& tensors,
                                           GgufTensorType qtype,
                                           const std::string& name = "weight");

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

// Map a ggml quant type name (e.g. "Q4_K") to its GgufTensorType id. Throws if unknown.
GgufTensorType gguf_type_from_name(const std::string& quant_type);

// One split part of a fused attn_qkv weight: its extracted tensors and shared quant type.
// Used by the GGUF builder to emit a
// GGML_OP_NONE weight leaf per q/k/v part (routing them through translate_weight like any other
// weight) instead of building the decompression nodes eagerly.
struct FusedQkvPart {
    WeightTensors tensors;
    GgufTensorType qtype = GGUF_TYPE_F16;
};

// Row-slice a fused `<base>` attn_qkv weight into q/k/v tensor payloads (no OV nodes).
std::array<FusedQkvPart, 3> split_fused_qkv_extracted(const std::string& base,
                                                      const std::unordered_map<std::string, ov::Tensor>& weights,
                                                      const std::unordered_map<std::string, GgufTensorType>& qtypes,
                                                      size_t n_q,
                                                      size_t n_k,
                                                      size_t n_v);

// Row-slice a fused `<base>.bias` (plain, unquantized 1D tensor) into q/k/v parts, using the
// same row ranges as split_fused_qkv_extracted. Callers only invoke this when `<base>.bias`
// is present -- not every fused-QKV arch has one.
std::array<ov::Tensor, 3> split_fused_qkv_bias(const std::string& base,
                                               const std::unordered_map<std::string, ov::Tensor>& weights,
                                               size_t n_q,
                                               size_t n_k,
                                               size_t n_v);

// De-interleave a qwen35 `<base>` attn_q weight, which packs the query and the attention output
// gate per head as [q_h0 | gate_h0 | q_h1 | gate_h1 | ...], into two plain projections.
// Returns {query, gate}.
std::array<FusedQkvPart, 2> split_interleaved_q_gate(const std::string& base,
                                                     const std::unordered_map<std::string, ov::Tensor>& weights,
                                                     const std::unordered_map<std::string, GgufTensorType>& qtypes,
                                                     size_t head_dim);

}  // namespace ov::frontend::gguf
