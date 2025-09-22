// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_dimension.hpp"

#include <ostream>

#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

std::ostream& operator<<(std::ostream& str, const StaticDimension& dimension) {
    return str << dimension.get_length();
}

StaticDimension::StaticDimension(value_type dimension) : m_dimension(dimension) {}

StaticDimension::StaticDimension(value_type ldimension, value_type udimension) : m_dimension(ldimension) {
    OPENVINO_ASSERT(ldimension == udimension,
                    "Can not create StaticDimension out of [",
                    ldimension,
                    ", ",
                    udimension,
                    "]");
}

StaticDimension::StaticDimension(const ov::Dimension& dim) {
    if (dim.is_static()) {
        m_dimension = dim.get_length();
    } else {
        // For dynamic dimensions, set to 0
        m_dimension = 0;
    }
}

bool StaticDimension::operator==(const StaticDimension& dim) const {
    return m_dimension == dim.m_dimension;
}

bool StaticDimension::operator!=(const StaticDimension& dim) const {
    return m_dimension != dim.m_dimension;
}

bool StaticDimension::operator!=(value_type val) const {
    return m_dimension != val;
}

bool StaticDimension::operator!=(int val) const {
    return m_dimension != static_cast<value_type>(val);
}

StaticDimension StaticDimension::operator+(const StaticDimension& dim) const {
    return {m_dimension + dim.m_dimension};
}

StaticDimension& StaticDimension::operator+=(const StaticDimension& dim) {
    return (*this = *this + dim);
}

StaticDimension StaticDimension::operator-(const StaticDimension& dim) const {
    return {m_dimension - dim.m_dimension};
}

StaticDimension StaticDimension::operator*(const StaticDimension& dim) const {
    return {m_dimension * dim.m_dimension};
}

StaticDimension StaticDimension::operator+(value_type val) const {
    return {m_dimension + val};
}

StaticDimension StaticDimension::operator-(value_type val) const {
    return {m_dimension - val};
}

StaticDimension StaticDimension::operator*(value_type val) const {
    return {m_dimension * val};
}

StaticDimension& StaticDimension::operator*=(const StaticDimension& dim) {
    return (*this = *this * dim);
}

StaticDimension StaticDimension::operator/(const value_type divisor) const {
    OPENVINO_ASSERT(divisor > 0, "divisor must be greater than 0");

    if (m_dimension % divisor) {
        return StaticDimension{};
    }
    return {m_dimension / divisor};
}

StaticDimension& StaticDimension::operator/=(const value_type divisor) {
    return (*this = *this / divisor);
}

StaticDimension StaticDimension::operator&(const StaticDimension& dim) const {
    return (*this == dim) ? dim : StaticDimension(0);
}

StaticDimension& StaticDimension::operator&=(const StaticDimension& dim) {
    if (*this != dim) {
        m_dimension = 0;
    }
    return *this;
}

bool StaticDimension::compatible(const StaticDimension& dim) const {
    return m_dimension == dim.m_dimension;
}

bool StaticDimension::compatible(value_type d) const {
    return m_dimension == d;
}

bool StaticDimension::same_scheme(const StaticDimension& dim) const {
    return m_dimension == dim.m_dimension;
}

bool StaticDimension::merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2) {
    if (d1 != d2) {
        return false;
    }
    dst = d1;
    return true;
}

bool StaticDimension::broadcast_merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2) {
    if (d1.get_length() == 1) {
        dst = d2;
        return true;
    }
    if (d2.get_length() == 1) {
        dst = d1;
        return true;
    }
    return merge(dst, d1, d2);
}

StaticDimension& StaticDimension::operator=(const ov::Dimension& dim) {
    if (dim.is_static()) {
        m_dimension = dim.get_length();
    } else {
        // For dynamic dimensions, set to 0 (or could throw)
        m_dimension = 0;
    }
    return *this;
}

StaticDimension& StaticDimension::operator=(value_type val) {
    m_dimension = val;
    return *this;
}

StaticDimension StaticDimension::operator*(const ov::Dimension& dim) const {
    if (dim.is_static()) {
        return {static_cast<value_type>(m_dimension * dim.get_length())};
    }
    // For dynamic dimensions, return 0 (or could throw)
    return {0};
}

bool StaticDimension::operator!=(const ov::Dimension& dim) const {
    if (dim.is_static()) {
        return m_dimension != static_cast<value_type>(dim.get_length());
    }
    // Static dimension is never equal to dynamic dimension
    return true;
}

bool StaticDimension::operator==(value_type val) const {
    return m_dimension == val;
}

bool StaticDimension::operator==(int val) const {
    return m_dimension == static_cast<value_type>(val);
}

bool StaticDimension::operator==(const ov::Dimension& dim) const {
    if (dim.is_static()) {
        return m_dimension == static_cast<value_type>(dim.get_length());
    }
    // Static dimension is never equal to dynamic dimension
    return false;
}

StaticDimension::value_type StaticDimension::get_length() const {
    return m_dimension;
}

StaticDimension::value_type StaticDimension::get_max_length() const {
    return m_dimension;
}

StaticDimension::value_type StaticDimension::get_min_length() const {
    return m_dimension;
}

}  // namespace ov::intel_cpu
