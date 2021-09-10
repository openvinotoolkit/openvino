// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/shape.hpp"
#include "ngraph/util.hpp"

using namespace std;

std::ostream& ov::operator<<(std::ostream& s, const StaticShape& shape) {
    s << "{";
    s << ngraph::join(shape);
    s << "}";
    return s;
}

ov::StaticShape::StaticShape() : std::vector<size_t>() {}

ov::StaticShape::StaticShape(const std::initializer_list<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::StaticShape::StaticShape(const std::vector<size_t>& axis_lengths) : std::vector<size_t>(axis_lengths) {}

ov::StaticShape::StaticShape(const StaticShape& axis_lengths) = default;

ov::StaticShape::StaticShape(size_t n, size_t initial_value) : std::vector<size_t>(n, initial_value) {}

ov::StaticShape::~StaticShape() = default;

ov::StaticShape& ov::StaticShape::operator=(const StaticShape& v) {
    static_cast<std::vector<size_t>*>(this)->operator=(v);
    return *this;
}

ov::StaticShape& ov::StaticShape::operator=(StaticShape&& v) noexcept {
    static_cast<std::vector<size_t>*>(this)->operator=(std::move(v));
    return *this;
}

constexpr ov::DiscreteTypeInfo ov::AttributeAdapter<ov::StaticShape>::type_info;
