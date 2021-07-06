// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Coordinate& coordinate)
{
    s << "Coordinate{";
    s << ngraph::join(coordinate);
    s << "}";
    return s;
}

ngraph::Coordinate::Coordinate() {}

ngraph::Coordinate::Coordinate(const std::initializer_list<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::Coordinate::Coordinate(const Shape& shape)
    : std::vector<size_t>(static_cast<const std::vector<size_t>&>(shape))
{
}

ngraph::Coordinate::Coordinate(const std::vector<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::Coordinate::Coordinate(const Coordinate& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::Coordinate::Coordinate(size_t n, size_t initial_value)
    : std::vector<size_t>(n, initial_value)
{
}

ngraph::Coordinate::~Coordinate() {}

ngraph::Coordinate& ngraph::Coordinate::operator=(const Coordinate& v)
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::Coordinate& ngraph::Coordinate::operator=(Coordinate&& v) noexcept
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

constexpr ngraph::DiscreteTypeInfo ngraph::AttributeAdapter<ngraph::Coordinate>::type_info;
