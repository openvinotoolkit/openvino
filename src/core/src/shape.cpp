// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape.hpp"

#include "ngraph/util.hpp"

using namespace std;

std::ostream& ov::operator<<(std::ostream& s, const Shape& shape) {
    s << "{";
    s << ngraph::join(shape);
    s << "}";
    return s;
}

ov::Shape::Shape() : std::vector<size_t>() {}

ov::Shape::Shape(const std::initializer_list<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::Shape::Shape(const std::vector<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::Shape::Shape(const Shape& axis_lengths) = default;

ov::Shape::Shape(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ov::Shape::~Shape() = default;

ov::Shape& ov::Shape::operator=(const Shape& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::Shape& ov::Shape::operator=(Shape&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

BWDCMP_RTTI_DEFINITION(ov::AttributeAdapter<ov::Shape>);
