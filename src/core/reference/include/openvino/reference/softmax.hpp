// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "openvino/core/shape_util.hpp"
#include "openvino/reference/reduce_max.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T>
void softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    const auto temp_shape = util::reduce_keep_dims(shape, axes);
    const auto temp_elements = shape_size(temp_shape);
    auto temp_storage = std::vector<T>(temp_elements);
    const auto temp_ptr = temp_storage.data();

    reduce_max(arg, temp_ptr, shape, axes);

    const CoordinateTransformBasic transform{shape};
    for (const auto& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] = std::exp(arg[out_index] - temp_ptr[temp_index]);
    }

    reduce_sum(out, temp_ptr, shape, axes);

    for (const auto& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] /= temp_ptr[temp_index];
    }
}
}  // namespace reference
}  // namespace ov
