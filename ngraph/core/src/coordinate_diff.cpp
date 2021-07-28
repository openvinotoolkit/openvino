// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const CoordinateDiff& coordinate_diff)
{
    s << "CoordinateDiff{";
    s << ngraph::join(coordinate_diff);
    s << "}";
    return s;
}

ngraph::CoordinateDiff::CoordinateDiff(const std::initializer_list<std::ptrdiff_t>& diffs)
    : std::vector<std::ptrdiff_t>(diffs)
{
}

ngraph::CoordinateDiff::CoordinateDiff(const std::vector<std::ptrdiff_t>& diffs)
    : std::vector<std::ptrdiff_t>(diffs)
{
}

ngraph::CoordinateDiff::CoordinateDiff(const CoordinateDiff& diffs)
    : std::vector<std::ptrdiff_t>(diffs)
{
}

ngraph::CoordinateDiff::CoordinateDiff(size_t n, std::ptrdiff_t initial_value)
    : std::vector<std::ptrdiff_t>(n, initial_value)
{
}

ngraph::CoordinateDiff::CoordinateDiff() {}

ngraph::CoordinateDiff::~CoordinateDiff() {}

ngraph::CoordinateDiff& ngraph::CoordinateDiff::operator=(const CoordinateDiff& v)
{
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
    return *this;
}

ngraph::CoordinateDiff& ngraph::CoordinateDiff::operator=(CoordinateDiff&& v) noexcept
{
    static_cast<std::vector<std::ptrdiff_t>*>(this)->operator=(v);
    return *this;
}

constexpr ngraph::DiscreteTypeInfo ngraph::AttributeAdapter<ngraph::CoordinateDiff>::type_info;
