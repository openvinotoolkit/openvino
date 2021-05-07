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
