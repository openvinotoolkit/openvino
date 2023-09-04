// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"

namespace ov {

std::size_t coordinate_index(const Coordinate& c, const Shape& s);

}  // namespace ov
