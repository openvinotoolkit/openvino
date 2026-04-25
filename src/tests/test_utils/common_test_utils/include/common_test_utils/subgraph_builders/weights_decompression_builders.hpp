// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <optional>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/node.hpp"

namespace ov {
namespace test {
namespace utils {

enum class DecompressionType {
    empty,   // no decompression
    scalar,  // decompression with scalar shape
    full     // decompression with per-channel or grouped shape
};

std::ostream& operator<<(std::ostream& os, DecompressionType type);

// fill the weights and scales with random values
std::shared_ptr<ov::Node> initMatMulDecompressionSubgraph(
    const ov::Shape& weights_shape,
    const int group_size,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const ov::element::Type decompression_precision,
    const ov::element::Type scale_precision,
    const bool transpose_weights,
    const DecompressionType decompression_multiply_type,
    const DecompressionType decompression_subtract_type,
    const bool reshape_on_decompression_constant,
    const std::optional<bool>& insert_transpose_node = std::nullopt,
    const size_t seed = 1);

// do real quantization of a random float tensor to get weights and scales.
// The optional `fp32_weights_gen_data` overrides the fp32 weight value range used before
// quantization (default: [1, 7] which gives dequantized magnitudes ~1.2..6 — large enough to
// overflow f16 in deep MoE chains; pass a centered range like (-1, 2, ...) for realistic
// quantized-LLM magnitudes).
std::shared_ptr<ov::Node> initMatMulDecompressionSubgraphQuantization(
    const ov::Shape& weights_shape,
    const int group_size,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const ov::element::Type decompression_precision,
    const ov::element::Type scale_precision,
    const bool transpose_weights,
    const DecompressionType decompression_multiply_type,
    const DecompressionType decompression_subtract_type,
    const bool reshape_on_decompression_constant,
    const std::optional<bool>& insert_transpose_node = std::nullopt,
    const size_t seed = 1,
    const std::optional<InputGenerateData>& fp32_weights_gen_data = std::nullopt);

std::shared_ptr<ov::Node> initGatherDecompressionSubgraph(const ov::Shape& data_shape,
                                                          const int group_size,
                                                          const ov::element::Type data_precision,
                                                          const ov::element::Type output_precision,
                                                          const bool add_subtract,
                                                          const bool reshape_on_decompression_constant,
                                                          const bool per_tensor_zp,
                                                          const bool per_tensor_scale);

}  // namespace utils
}  // namespace test
}  // namespace ov