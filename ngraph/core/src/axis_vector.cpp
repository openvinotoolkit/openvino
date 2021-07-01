// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/axis_vector.hpp"
#include "ngraph/util.hpp"

std::ostream& ngraph::operator<<(std::ostream& s, const AxisVector& axis_vector)
{
    s << "AxisVector{";
    s << ngraph::join(axis_vector);
    s << "}";
    return s;
}

ngraph::AxisVector::AxisVector(const std::initializer_list<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::AxisVector::AxisVector(const std::vector<size_t>& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::AxisVector::AxisVector(const AxisVector& axes)
    : std::vector<size_t>(axes)
{
}

ngraph::AxisVector::AxisVector(size_t n)
    : std::vector<size_t>(n)
{
}

ngraph::AxisVector::AxisVector() {}

ngraph::AxisVector::~AxisVector() {}

ngraph::AxisVector& ngraph::AxisVector::operator=(const AxisVector& v)
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::AxisVector& ngraph::AxisVector::operator=(AxisVector&& v) noexcept
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

constexpr ngraph::DiscreteTypeInfo ngraph::AttributeAdapter<ngraph::AxisVector>::type_info;
