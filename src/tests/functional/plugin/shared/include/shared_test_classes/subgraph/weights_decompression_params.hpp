// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

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

}  // namespace test
}  // namespace ov
