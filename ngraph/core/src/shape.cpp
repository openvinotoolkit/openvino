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

#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& s, const Shape& shape)
{
    s << "Shape{";
    s << ngraph::join(shape);
    s << "}";
    return s;
}

ngraph::Shape::Shape()
    : std::vector<size_t>()
{
}

ngraph::Shape::Shape(const std::initializer_list<size_t>& axis_lengths)
    : std::vector<size_t>(axis_lengths)
{
}

ngraph::Shape::Shape(const std::vector<size_t>& axis_lengths)
    : std::vector<size_t>(axis_lengths)
{
}

ngraph::Shape::Shape(const Shape& axis_lengths)
    : std::vector<size_t>(axis_lengths)
{
}

ngraph::Shape::Shape(size_t n, size_t initial_value)
    : std::vector<size_t>(n, initial_value)
{
}

ngraph::Shape::~Shape() {}

ngraph::Shape& ngraph::Shape::operator=(const Shape& v)
{
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ngraph::Shape& ngraph::Shape::operator=(Shape&& v) noexcept
{
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

constexpr DiscreteTypeInfo AttributeAdapter<Shape>::type_info;
