// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate_index.hpp"

#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
std::size_t coordinate_index(const Coordinate& c, const Shape& s) {
    if (c.size() < s.size()) {
        throw std::domain_error("Coordinate rank is less than shape rank.");
    }
    std::size_t index = 0;
    std::size_t stride = 1;
    std::size_t const padding = c.size() - s.size();

    for (std::size_t axis = s.size(); axis-- > 0;) {
        if (s[axis] > 1) {
            index += c[axis + padding] * stride;
            stride *= s[axis];
        }
    }

    return index;
}
}  // namespace ngraph
