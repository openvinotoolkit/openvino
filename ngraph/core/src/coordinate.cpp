// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ov;

std::ostream& ov::operator<<(std::ostream& s, const Coordinate& coordinate)
{
    s << "Coordinate{";
    s << ov::join(coordinate);
    s << "}";
    return s;
}

ov::Coordinate::Coordinate() {}

ov::Coordinate::Coordinate(const std::initializer_list<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ov::Coordinate::Coordinate(const Shape& shape)
    : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape))
{
}

ov::Coordinate::Coordinate(const std::vector<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ov::Coordinate::Coordinate(const Coordinate& axes)
    : std::vector<size_t>(axes)
{
}

ov::Coordinate::Coordinate(size_t n, size_t initial_value)
    : std::vector<size_t>(n, initial_value)
{
}

ov::Coordinate::~Coordinate() {}

ov::Coordinate& ov::Coordinate::operator=(const Coordinate& v)
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::Coordinate& ov::Coordinate::operator=(Coordinate&& v) noexcept
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ov::Coordinate>::type_info;
