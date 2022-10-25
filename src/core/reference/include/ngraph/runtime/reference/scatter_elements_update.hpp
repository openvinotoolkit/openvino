// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstring>

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename DataType, typename IndicesType>
void scatter_elem_update(const DataType* input_data,
                         const IndicesType* indices,
                         const DataType* updates,
                         const int64_t& axis,
                         DataType* out_buf,
                         const Shape& data_shape,
                         const Shape& indices_shape) {
    // Copy inputs to out
    std::memcpy(out_buf, input_data, sizeof(DataType) * shape_size(data_shape));

    // 3D example
    // output[indices[i][j][k]][j][k] = updates[i][j][k] if axis = 0,
    // output[i][indices[i][j][k]][k] = updates[i][j][k] if axis = 1,
    // output[i][j][indices[i][j][k]] = updates[i][j][k] if axis = 2

    CoordinateTransformBasic indices_transform{indices_shape};
    CoordinateTransformBasic data_transform{data_shape};
    const auto indices_strides = row_major_strides(indices_shape);
    const auto data_strides = row_major_strides(data_shape);

    for (const Coordinate& indices_cord : indices_transform) {
        const size_t indices_idx =
            std::inner_product(indices_cord.begin(), indices_cord.end(), indices_strides.begin(), uint64_t(0));
        Coordinate out_cord(indices_cord);
        out_cord.at(axis) = indices[indices_idx];
        const auto out_idx = std::inner_product(out_cord.begin(), out_cord.end(), data_strides.begin(), uint64_t(0));
        out_buf[out_idx] = updates[indices_idx];
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
