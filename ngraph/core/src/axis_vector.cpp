//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

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
