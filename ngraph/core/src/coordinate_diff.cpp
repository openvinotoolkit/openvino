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
