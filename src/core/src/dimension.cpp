// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/dimension.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>

#include "dimension_tracker.hpp"

using namespace ngraph;

std::ostream& ov::operator<<(std::ostream& str, const Dimension& dimension) {
    if (dimension.is_static()) {
        return str << dimension.get_length();
    } else if (dimension.get_min_length() > 0) {
        str << dimension.get_min_length() << "..";
        if (dimension.get_interval().has_upper_bound())
            return str << dimension.get_max_length();
        else
            return str;
    } else if (dimension.get_interval().has_upper_bound()) {
        return str << ".." << dimension.get_max_length();
    } else {
        return str << "?";
    }
}

Dimension::Dimension(value_type dimension)
    : m_dimension(dimension == -1 ? 0 : dimension, dimension == -1 ? Interval::s_max : dimension) {}

Dimension::Dimension(value_type min_dimension, value_type max_dimension)
    : m_dimension(min_dimension == -1 ? 0 : min_dimension, max_dimension == -1 ? Interval::s_max : max_dimension) {}

Dimension Dimension::operator+(const Dimension& dim) const {
    if (dim.m_dimension == 0)
        return *this;
    else if (m_dimension == 0)
        return dim;
    return Dimension(m_dimension + dim.m_dimension);
}

Dimension Dimension::operator-(const Dimension& dim) const {
    if (dim.m_dimension == 0)
        return *this;
    return Dimension(m_dimension - dim.m_dimension);
}

Dimension Dimension::operator/(const value_type divisor) const {
    OPENVINO_ASSERT(divisor >= 0, "divisor must be greater than 0");
    if (divisor == 1)
        return *this;
    if (m_dimension.get_max_val() == Interval::s_max && m_dimension.get_min_val() == 0)
        return Dimension::dynamic();
    const auto& lower_bound = ceil(static_cast<double>(m_dimension.get_min_val()) / divisor);
    const auto& upper_bound = floor(static_cast<double>(m_dimension.get_max_val()) / divisor);
    return Dimension(lower_bound, upper_bound);
}

Dimension Dimension::operator*(const Dimension& dim) const {
    if (dim.m_dimension == 1)
        return *this;
    else if (m_dimension == 1)
        return dim;
    return Dimension(m_dimension * dim.m_dimension);
}

Dimension Dimension::operator&(const Dimension& dim) const {
    return Dimension(m_dimension & dim.m_dimension);
}

Dimension& Dimension::operator&=(const Dimension& dim) {
    m_dimension &= dim.m_dimension;
    return *this;
}

bool Dimension::compatible(const Dimension& d) const {
    return !(m_dimension & d.m_dimension).empty();
}

bool Dimension::relaxes(const Dimension& d) const {
    return m_dimension.contains(d.m_dimension);
}

bool Dimension::refines(const Dimension& d) const {
    return d.m_dimension.contains(m_dimension);
}

bool Dimension::same_scheme(const Dimension& dim) const {
    return (m_dimension == dim.m_dimension) || (m_dimension.size() > 1 && dim.m_dimension.size() > 1);
}

bool Dimension::merge(Dimension& dst, const Dimension& d1, const Dimension& d2) {
    auto result = d1.m_dimension & d2.m_dimension;
    if (result.empty()) {
        return false;
    }
    dst = result;

    if (auto& t = d1.m_table_of_equivalence)
        t->set_as_equal(d1, d2);
    else if (auto& t = d2.m_table_of_equivalence)
        t->set_as_equal(d1, d2);
    if (d1.m_label == d2.m_label || d2.m_label == 0)
        dst.m_label = d1.m_label;
    else if (d1.m_label == 0)
        dst.m_label = d2.m_label;
    return true;
}

bool Dimension::broadcast_merge(Dimension& dst, const Dimension& d1, const Dimension& d2) {
    bool d1_has_1 = d1.m_dimension.contains(1);
    bool d2_has_1 = d2.m_dimension.contains(1);
    if (d1_has_1 && d2_has_1) {
        auto result = ov::Interval(std::min(d1.m_dimension.get_min_val(), d2.m_dimension.get_min_val()),
                                   std::max(d1.m_dimension.get_max_val(), d2.m_dimension.get_max_val()));
        if (result.empty())
            return false;
        dst = Dimension(result);
        if (d1.m_label == d2.m_label || d2.m_label == 0)
            dst.m_label = d1.m_label;
        else if (d1.m_label == 0)
            dst.m_label = d2.m_label;
        return true;
    } else if (d1_has_1) {
        dst = d2;
    } else if (d2_has_1) {
        dst = d1;
    } else {
        return merge(dst, d1, d2);
    }
    return true;
}

Dimension::value_type Dimension::get_length() const {
    if (is_dynamic()) {
        throw std::invalid_argument("Cannot get length of dynamic dimension");
    }
    return m_dimension.get_min_val();
}

namespace {
Dimension::value_type dimension_length(Interval::value_type vt) {
    return vt == Interval::s_max ? -1 : vt;
}
}  // namespace

Dimension::value_type Dimension::get_max_length() const {
    return dimension_length(m_dimension.get_max_val());
}

Dimension::value_type Dimension::get_min_length() const {
    return dimension_length(m_dimension.get_min_val());
}
