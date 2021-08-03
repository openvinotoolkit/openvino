// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ngraph
{
    class Coordinate;
    class Shape;
} // namespace ngraph

namespace ngraph
{
    std::size_t coordinate_index(const Coordinate& c, const Shape& s);
}
