// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/runtime/tensor.hpp"
namespace ov {
class MappedMemory;
}

namespace ov::frontend::gguf {

// GGUF tensor (quantization) type ids, matching the on-disk GGUF format numbering.
enum GgufTensorType {
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
    GGUF_TYPE_Q1_0 = 41,   // binary: f16 scale + 128x 1-bit codes, value = bit ? +scale : -scale
    GGUF_TYPE_Q2_0 = 42,   // ternary: f16 scale + 64x 2-bit codes, value = (code - 1) * scale
    GGUF_TYPE_PQ2_0 = 142,
    GGUF_TYPE_PTQ1_0 = 143,
    GGUF_TYPE_COUNT,
};

// GGUF metadata value type ids (the kv-pair value encoding).
enum GgufValueType {
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

// A metadata value: scalars are stored as an ov::Tensor of shape {} (so that numeric
// metadata round-trips through ov::element types), strings as std::string, and arrays as
// an ov::Tensor of shape {n} or a vector<std::string>.
using GGUFMetaData =
    std::variant<std::monostate, float, int, ov::Tensor, std::string, std::vector<std::string>, std::vector<int32_t>>;

// Parsed arrays own repacked quantized buffers and keep non-quantized mmap views alive.
using GGUFLoad = std::tuple<std::unordered_map<std::string, GGUFMetaData>,
                            std::unordered_map<std::string, ov::Tensor>,
                            std::unordered_map<std::string, GgufTensorType>,
                            std::shared_ptr<ov::MappedMemory>>;

/// \brief Parse GGUF metadata and optionally tensors. Each repacked quantized tensor owns its buffer.
/// Non-quantized tensors share views into the memory-mapped file kept alive by the returned MappedMemory.
/// \param file Path to the GGUF file.
/// \param load_weights If false, parses only header and metadata KV pairs, returning empty tensor maps.
///                     If true (default), also loads and repacks all tensor weights.
GGUF_FRONTEND_API GGUFLoad get_gguf_data(const std::string& file, bool load_weights = true);

}  // namespace ov::frontend::gguf
