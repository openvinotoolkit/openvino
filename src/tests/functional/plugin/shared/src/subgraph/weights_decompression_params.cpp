// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/weights_decompression_params.hpp"

namespace ov {
namespace test {
std::ostream& operator<<(std::ostream& os, MatMulDecompressionShapeParams shape_params) {
    os << "data_shape=" << shape_params.data_shape << "_weights_shape=" << shape_params.weights_shape;
    if (shape_params.decompression_group_size != -1)
        os << "_group_size=" << shape_params.decompression_group_size;
    return os;
}

std::ostream& operator<<(std::ostream& os, GatherDecompressionShapeParams shape_params) {
    os << "data_shape=" << shape_params.data_shape << "_indices_shape=" << shape_params.indices_shape;
    if (shape_params.decompression_group_size != -1)
        os << "_group_size=" << shape_params.decompression_group_size;
    os << "_axis=" << shape_params.axis << "_batch_dims=" << shape_params.batch_dims;
    return os;
}

}  // namespace test
}  // namespace ov
