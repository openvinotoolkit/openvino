// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov
{
    class Coordinate;
    class Shape;
} // namespace ov

namespace ov
{
    std::size_t coordinate_index(const Coordinate& c, const Shape& s);
}
