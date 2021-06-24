// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <iostream>
#include <limits>
#include <sstream>

#include "ngraph/dimension.hpp"

using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& str, const Dimension& dimension)
{
    if (dimension.is_static())
    {
        return str << dimension.get_length();
    }
    else if (dimension.get_interval().has_upper_bound())
    {
        return str << "[" << dimension.get_min_length() << ", " << dimension.get_max_length()
                   << "]";
    }
    else
    {
        return str << "?";
    }
}

Dimension::Dimension(value_type dimension)
    : m_dimension(dimension == -1 ? 0 : dimension, dimension == -1 ? Interval::s_max : dimension)
{
}

Dimension::Dimension(value_type min_dimension, value_type max_dimension)
    : m_dimension(min_dimension == -1 ? 0 : min_dimension,
                  max_dimension == -1 ? Interval::s_max : max_dimension)
{
}

Dimension Dimension::operator+(const Dimension& dim) const
{
    return Dimension(m_dimension + dim.m_dimension);
}

Dimension Dimension::operator-(const Dimension& dim) const
{
    return Dimension(m_dimension - dim.m_dimension);
}

Dimension Dimension::operator*(const Dimension& dim) const
{
    return Dimension(m_dimension * dim.m_dimension);
}

Dimension Dimension::operator&(const Dimension& dim) const
{
    return Dimension(m_dimension & dim.m_dimension);
}

Dimension& Dimension::operator&=(const Dimension& dim)
{
    m_dimension &= dim.m_dimension;
    return *this;
}

bool Dimension::compatible(const Dimension& d) const
{
    return !(m_dimension & d.m_dimension).empty();
}

bool Dimension::relaxes(const Dimension& d) const
{
    return m_dimension.contains(d.m_dimension);
}

bool Dimension::refines(const Dimension& d) const
{
    return d.m_dimension.contains(m_dimension);
}

bool Dimension::same_scheme(const Dimension& dim) const
{
    return (m_dimension == dim.m_dimension) ||
           (m_dimension.size() > 1 && dim.m_dimension.size() > 1);
}

bool Dimension::merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    auto result = d1.m_dimension & d2.m_dimension;
    if (result.empty())
    {
        return false;
    }
    dst = result;
    return true;
}

bool Dimension::broadcast_merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    if (d1.m_dimension.size() == 1 && d1.m_dimension.get_min_val() == 1)
    {
        dst = d2;
        return true;
    }
    if (d2.m_dimension.size() == 1 && d2.m_dimension.get_min_val() == 1)
    {
        dst = d1;
        return true;
    }
    return merge(dst, d1, d2);
}

Dimension::value_type Dimension::get_length() const
{
    if (is_dynamic())
    {
        throw std::invalid_argument("Cannot get length of dynamic dimension");
    }
    return m_dimension.get_min_val();
}

namespace
{
    Dimension::value_type dimension_length(Interval::value_type vt)
    {
        return vt == Interval::s_max ? -1 : vt;
    }
} // namespace

Dimension::value_type Dimension::get_max_length() const
{
    return dimension_length(m_dimension.get_max_val());
}

Dimension::value_type Dimension::get_min_length() const
{
    return dimension_length(m_dimension.get_min_val());
}
