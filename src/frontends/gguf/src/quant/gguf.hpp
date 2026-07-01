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

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// GGUF tensor (quantization) type ids, matching the on-disk GGUF format numbering.
// Only the subset the frontend handles is enumerated explicitly; others are accepted
// numerically but rejected at dequant time.
enum gguf_tensor_type {
    GGUF_TYPE_F32 = 0,
    GGUF_TYPE_F16 = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q5_0 = 6,
    GGUF_TYPE_Q5_1 = 7,
    GGUF_TYPE_Q8_0 = 8,
    GGUF_TYPE_Q8_1 = 9,
    GGUF_TYPE_Q2_K = 10,
    GGUF_TYPE_Q3_K = 11,
    GGUF_TYPE_Q4_K = 12,
    GGUF_TYPE_Q5_K = 13,
    GGUF_TYPE_Q6_K = 14,
    GGUF_TYPE_Q8_K = 15,
    GGUF_TYPE_I8 = 24,
    GGUF_TYPE_I16 = 25,
    GGUF_TYPE_I32 = 26,
    GGUF_TYPE_I64 = 27,
    GGUF_TYPE_F64 = 28,
    GGUF_TYPE_BF16 = 30,
    GGUF_TYPE_MXFP4 = 39,  // 4-bit microscaling (gpt-oss): 1-byte E8M0 scale + 32x E2M1
    GGUF_TYPE_COUNT,
};

// GGUF metadata value type ids (the kv-pair value encoding).
enum gguf_value_type {
    GGUF_VALUE_TYPE_UINT8 = 0,
    GGUF_VALUE_TYPE_INT8 = 1,
    GGUF_VALUE_TYPE_UINT16 = 2,
    GGUF_VALUE_TYPE_INT16 = 3,
    GGUF_VALUE_TYPE_UINT32 = 4,
    GGUF_VALUE_TYPE_INT32 = 5,
    GGUF_VALUE_TYPE_FLOAT32 = 6,
    GGUF_VALUE_TYPE_BOOL = 7,
    GGUF_VALUE_TYPE_STRING = 8,
    GGUF_VALUE_TYPE_ARRAY = 9,
    GGUF_VALUE_TYPE_UINT64 = 10,
    GGUF_VALUE_TYPE_INT64 = 11,
    GGUF_VALUE_TYPE_FLOAT64 = 12,
};

// A single tensor descriptor parsed from the GGUF file. `weights_data` points directly
// into the memory-mapped file (zero-copy); the owning reader keeps the mapping alive.
// Field names mirror the classic gguf-tools layout so the dequant code in
// gguf_quants.cpp reads them unchanged.
struct gguf_tensor {
    const char* name = nullptr;  // points into the mmap (not null-terminated)
    size_t namelen = 0;
    uint32_t type = 0;  // gguf_tensor_type
    uint32_t ndim = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t offset = 0;       // offset from the start of the tensor-data section
    uint64_t bsize = 0;        // total size in bytes
    uint64_t num_weights = 0;  // total number of elements
    const uint8_t* weights_data = nullptr;
};

// A metadata value: scalars are stored as an ov::Tensor of shape {} (so that numeric
// metadata round-trips through ov::element types), strings as std::string, and arrays as
// an ov::Tensor of shape {n} or a vector<std::string>.
using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

// GGUFLoad result: (metadata, tensor arrays, qtype map, mmap, quant_buf).
// - mmap: must stay alive while arrays tensors are used (non-quantized tensors are mmap views).
// - quant_buf: single AlignedBuffer holding all repacked quantized weight/scale/bias data;
//   tensors in `arrays` for quantized weights are SharedBuffer slices into this buffer.
using GGUFLoad = std::tuple<std::unordered_map<std::string, GGUFMetaData>,
                            std::unordered_map<std::string, ov::Tensor>,
                            std::unordered_map<std::string, gguf_tensor_type>,
                            std::shared_ptr<ov::MappedMemory>,
                            std::shared_ptr<ov::AlignedBuffer>>;

// Reverse of the GGML dimension order (GGUF stores dims fastest-first).
ov::Shape get_shape(const gguf_tensor& tensor);

// Fill pre-allocated i4 weights (u32-packed, XORed for i4 sign) and f16 scales from a
// Q4_0 tensor. No bias: Q4_0 is symmetric (zp = -8*scale is implicit, not stored).
void gguf_fill_q4_0(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales);

// Fill pre-allocated weights and f16 scales from a symmetric GGUF tensor
// (Q8_0/Q5_0/Q6_K: i8 weights; Q3_K: i4 weights packed as u8).
// No zero-point: the center value is subtracted during unpacking so weights are centered at 0.
void gguf_fill_sym(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales);

// Fill pre-allocated weights, f16 scales, and integer zero-points from an asymmetric GGUF
// tensor (Q4_1: u4 zp; Q4_K: u4 zp; Q5_K/Q5_1: u8 zp; Q2_K: u8 zp).
// Tensor shapes must match quant_sizes.
void gguf_fill_asym(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales, ov::Tensor& zp);

// Fill pre-allocated f4e2m1 weights and f8e8m0 scales from an MXFP4 GGUF tensor.
void gguf_fill_mxfp4(const gguf_tensor& tensor, ov::Tensor& weights, ov::Tensor& scales);

// Parse a GGUF file: returns (metadata, tensors-by-ggml-name, qtype map, mmap, quant_buf).
// Non-quantized tensors are zero-copy views into the mmap (mmap must outlive arrays use).
// Quantized tensors are SharedBuffer slices of a single AlignedBuffer (quant_buf) so all
// repacked weight/scale/bias data lives in one allocation (IR-frontend pattern).
GGUFLoad get_gguf_data(const std::string& file);

// Extract the architecture config (architecture, layer_num, head_num, head_size,
// head_num_kv, hidden_size, max_position_embeddings, rms_norm_eps, rope_freq_base,
// file_type) from parsed metadata.
std::map<std::string, GGUFMetaData> config_from_meta(const std::unordered_map<std::string, GGUFMetaData>& metadata);

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
