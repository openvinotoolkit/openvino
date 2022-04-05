// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ngraph/coordinate.hpp>
#include <ngraph/shape.hpp>

namespace ngraph {

std::size_t coordinate_index(const Coordinate& c, const Shape& s);

}  // namespace ngraph
