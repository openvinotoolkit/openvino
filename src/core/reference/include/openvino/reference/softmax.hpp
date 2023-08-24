// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"
#include "openvino/reference/max.hpp"
#include "openvino/reference/sum.hpp"

namespace ov {
namespace reference {
template <typename T>
void softmax(const T* arg, T* out, const Shape& shape, const AxisSet& axes) {
    NGRAPH_SUPPRESS_DEPRECATED_START
    auto temp_shape = ngraph::reduce(shape, axes, true);
    auto temp_elements = shape_size(temp_shape);
    auto temp_ptr = new T[temp_elements];

    ngraph::runtime::reference::max(arg, temp_ptr, shape, axes);

    ngraph::CoordinateTransform transform(shape);
    ngraph::CoordinateTransform temp_transform(temp_shape);
    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = ngraph::reduce(coord, axes, true);
        out[transform.index(coord)] =
            std::exp(arg[transform.index(coord)] - temp_ptr[temp_transform.index(temp_coord)]);
    }

    ngraph::runtime::reference::sum(out, temp_ptr, shape, axes);

    for (const Coordinate& coord : transform) {
        Coordinate temp_coord = ngraph::reduce(coord, axes, true);
        out[transform.index(coord)] /= temp_ptr[temp_transform.index(temp_coord)];
    }

    delete[] temp_ptr;
    NGRAPH_SUPPRESS_DEPRECATED_END
}
}  // namespace reference
}  // namespace ov
