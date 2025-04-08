// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"

namespace ov {

std::size_t coordinate_index(const Coordinate& c, const Shape& s);

/**
 * @brief Calculate offset from begin of buffer based on coordinate and strides.
 *
 * If coordinates and strides have different sizes then result is undefined behaviour.
 *
 * @param coordinate Vector with multi-dimension coordinates.
 * @param strides    Vector with multi-dimension strides
 * @return           Offset of element from start of buffer.
 */
size_t coordinate_offset(const std::vector<size_t>& coordinate, const std::vector<size_t>& strides);

}  // namespace ov
