// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <memory>
#include <optional>

#include "openvino/core/node.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {
struct MatMulDecompressionShapeParams {
    MatMulDecompressionShapeParams() = default;
    MatMulDecompressionShapeParams(InputShape data_shape, ov::Shape weights_shape, int decompression_group_size = -1)
        : data_shape(std::move(data_shape)),
          weights_shape(std::move(weights_shape)),
          decompression_group_size(decompression_group_size) {}

    InputShape data_shape;
    ov::Shape weights_shape;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int decompression_group_size;
};

std::ostream& operator<<(std::ostream& os, MatMulDecompressionShapeParams type);

struct GatherDecompressionShapeParams {
    GatherDecompressionShapeParams() = default;
    GatherDecompressionShapeParams(ov::Shape data_shape,
                                   InputShape indices_shape,
                                   int axis,
                                   int64_t batch_dims,
                                   int decompression_group_size = -1)
        : data_shape(std::move(data_shape)),
          indices_shape(std::move(indices_shape)),
          axis(axis),
          batch_dims(batch_dims),
          decompression_group_size(decompression_group_size) {}

    ov::Shape data_shape;
    InputShape indices_shape;
    int axis;
    int64_t batch_dims;
    // Decompression group size. If the value is equal to -1, ordinary decompression is used
    int decompression_group_size;
};

std::ostream& operator<<(std::ostream& os, GatherDecompressionShapeParams type);

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

// do real quantization of a random float tensor to get weights and scales
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
    const size_t seed = 1);

std::shared_ptr<ov::Node> initGatherDecompressionSubgraph(const ov::Shape& data_shape,
                                                          const int group_size,
                                                          const ov::element::Type data_precision,
                                                          const ov::element::Type output_precision,
                                                          const bool add_subtract,
                                                          const bool reshape_on_decompression_constant,
                                                          const bool per_tensor_zp,
                                                          const bool per_tensor_scale);

}  // namespace test
}  // namespace ov
