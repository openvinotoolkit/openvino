// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "static_dimension.hpp"

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

bool StaticDimension::operator==(const StaticDimension& dim) const {
    return m_dimension == dim.m_dimension;
}

bool StaticDimension::operator!=(const StaticDimension& dim) const {
    return m_dimension != dim.m_dimension;
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
    return (*this == dim) ? dim : 0;
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
    if (d1 == 1) {
        dst = d2;
        return true;
    }
    if (d2 == 1) {
        dst = d1;
        return true;
    }
    return merge(dst, d1, d2);
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
