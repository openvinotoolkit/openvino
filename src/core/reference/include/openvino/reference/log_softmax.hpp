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
void log_softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    const auto temp_shape = util::reduce_keep_dims(shape, axes);
    const auto temp_elements = shape_size(temp_shape);
    auto temp_max = std::vector<T>(temp_elements, 0);
    auto temp_sum = std::vector<T>(temp_elements, 0);

    reduce_max(arg, temp_max.data(), shape, axes);

    const CoordinateTransformBasic transform{shape};
    for (const Coordinate& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] = static_cast<T>(std::exp(arg[out_index] - temp_max[temp_index]));
    }

    reduce_sum(out, temp_sum.data(), shape, axes);

    for (const Coordinate& coord : transform) {
        const Coordinate temp_coord = util::reduce_keep_dims(coord, axes);
        const auto out_index = coordinate_index(coord, shape);
        const auto temp_index = coordinate_index(temp_coord, temp_shape);
        out[out_index] = static_cast<T>((arg[out_index] - temp_max[temp_index]) - std::log(temp_sum[temp_index]));
    }
}
}  // namespace reference
}  // namespace ov
