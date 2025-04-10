// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/coordinate_index.hpp"

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
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

size_t coordinate_offset(const std::vector<size_t>& coordinate, const std::vector<size_t>& strides) {
    return std::inner_product(coordinate.cbegin(), coordinate.cend(), strides.cbegin(), static_cast<size_t>(0));
}
}  // namespace ov
