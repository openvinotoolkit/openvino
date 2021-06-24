// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
