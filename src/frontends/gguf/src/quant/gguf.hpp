// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "openvino/frontend/gguf/parser.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov::frontend::gguf {

// A single tensor descriptor parsed from the GGUF file. `weights_data` points directly
// into the memory-mapped file (zero-copy); the owning reader keeps the mapping alive.
// Field names mirror the classic gguf-tools layout so the dequant code in
// gguf_quants.cpp reads them unchanged.
struct GgufTensor {
    const char* name = nullptr;  // points into the mmap (not null-terminated)
    size_t namelen = 0;
    uint32_t type = 0;  // GgufTensorType
    uint32_t ndim = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t offset = 0;       // offset from the start of the tensor-data section
    uint64_t bsize = 0;        // total size in bytes
    uint64_t num_weights = 0;  // total number of elements
    const uint8_t* weights_data = nullptr;
};

// Fill pre-allocated weights and f16 scales from a symmetric GGUF tensor
// (Q8_0/Q5_0/Q6_K: i8 weights; Q4_0/Q3_K: i4 weights; Q1_0: u1 weights).
// Integer weights are centered during unpacking; Q1_0 keeps binary codes and applies 2 * bit - 1 in the graph.
void gguf_fill_sym(const GgufTensor& tensor, ov::Tensor& weights, ov::Tensor& scales);

// Fill pre-allocated weights, f16 scales, and integer zero-points from an asymmetric GGUF
// tensor (Q4_1: u4 zp; Q4_K: u4 zp; Q5_K/Q5_1: u8 zp; Q2_K: u8 zp).
// Tensor shapes must match quant_sizes.
void gguf_fill_asym(const GgufTensor& tensor, ov::Tensor& weights, ov::Tensor& scales, ov::Tensor& zp);

// Fill pre-allocated f4e2m1 weights and f8e8m0 scales from an MXFP4 GGUF tensor.
void gguf_fill_mxfp4(const GgufTensor& tensor, ov::Tensor& weights, ov::Tensor& scales);

// Fill pre-allocated u2 weights, f16 scales and u8 zero-points from a Q2_0 (ternary) tensor.
// The zero-point is the constant 1 for every block: value = (code - 1) * scale.
void gguf_fill_q2_0(const GgufTensor& tensor, ov::Tensor& weights, ov::Tensor& scales, ov::Tensor& zp);

// Quantize one row to Q8_0_C: a single channel-wise f16 scale (amax/127) plus signed int8
// weights. Shared by every channel-wise requant source so the rounding and the zero-row rule
// live in one place.
void quantize_row_q8_0_c(const float* x, size_t cols, int8_t* out_weights, ov::float16& out_scale);

// Fused bit-exact ggml dequant + channel-wise Q8_0_C requant for the token_embd/output/Q6_K/Q5_K
// requant path. Streams one row at a time (never materializes the full f32 weight). Fills i8
// weights [rows,cols] + f16 scales [rows,1]; matches upstream's to_float->quantize_q8_0 exactly so
// those tensors are bit-identical to the vendored backend. Returns false for unsupported qtypes.
bool requantize_q8_0_channelwise_faithful(const GgufTensor& tensor,
                                          size_t rows,
                                          size_t cols,
                                          GgufTensorType qtype,
                                          int8_t* out_weights,
                                          ov::float16* out_scales);

// Test-only accessors for the per-row faithful K-quant dequant (see test_dequant_vs_ggml.cpp).
void dequant_row_q4_k_f32_for_test(const uint8_t* row, size_t cols, float* y);
void dequant_row_q5_k_f32_for_test(const uint8_t* row, size_t cols, float* y);
void dequant_row_q6_k_f32_for_test(const uint8_t* row, size_t cols, float* y);

// Parse GGUF metadata and tensors. Each repacked quantized tensor owns its buffer.
GGUFLoad get_gguf_data(const std::string& file);

// Extract the DECODER-family architecture config (architecture, layer_num, head_num, head_size,
// head_num_kv, hidden_size, max_position_embeddings, rms_norm_eps, rope_freq_base, file_type, ...)
// from parsed metadata.
//
// Every key it reads is prefixed with the LLM architecture name ("<arch>.block_count",
// "<arch>.attention.head_count", ...), so it is only meaningful for a causal-decoder GGUF. Call
// detect_model_kind() first: an mmproj file names its architecture "clip" and carries "clip.*"
// keys instead, and would fail here on a missing block_count. A future non-decoder family gets its
// own reader next to this one rather than extending it.
std::map<std::string, GGUFMetaData> decoder_config_from_meta(
    const std::unordered_map<std::string, GGUFMetaData>& metadata);

// Reverse of the GGML dimension order (GGUF stores dims fastest-first).
ov::Shape get_shape(const GgufTensor& tensor);

}  // namespace ov::frontend::gguf
