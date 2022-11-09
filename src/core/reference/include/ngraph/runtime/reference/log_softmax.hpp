// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/max.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
template <typename T>
void log_softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto temp_shape = reduce(shape, axes, true);
    auto temp_elements = shape_size(temp_shape);
    auto temp_max = std::vector<T>(temp_elements, 0);
    auto temp_sum = std::vector<T>(temp_elements, 0);

    max(arg, temp_max.data(), shape, axes);

    CoordinateTransform transform(shape);
    CoordinateTransform temp_transform(temp_shape);
    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = reduce(coord, axes, true);
        out[transform.index(coord)] =
            static_cast<T>(std::exp(arg[transform.index(coord)] - temp_max[temp_transform.index(temp_coord)]));
    }

    sum(out, temp_sum.data(), shape, axes);

    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = reduce(coord, axes, true);
        out[transform.index(coord)] =
            static_cast<T>((arg[transform.index(coord)] - temp_max[temp_transform.index(temp_coord)]) -
                           std::log(temp_sum[temp_transform.index(temp_coord)]));
    }
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
