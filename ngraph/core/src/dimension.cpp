// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "ngraph/dimension.hpp"

using namespace ngraph;

std::ostream& ngraph::operator<<(std::ostream& str, const Dimension& dimension)
{
    if (!dimension.get_name().empty())
        str << dimension.get_name() << ":";

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

Dimension::Dimension(value_type dimension, std::string name)
    : m_dimension(dimension == -1 ? 0 : dimension, dimension == -1 ? Interval::s_max : dimension)
    , m_name(name)
{
}

Dimension::Dimension(value_type min_dimension, value_type max_dimension, std::string name)
    : m_dimension(min_dimension == -1 ? 0 : min_dimension,
                  max_dimension == -1 ? Interval::s_max : max_dimension)
    , m_name(name)
{
}

Dimension Dimension::operator+(const Dimension& dim) const
{
    if (dim.m_dimension == 0 && dim.get_name().empty())
        return *this;
    else if (m_dimension == 0 && get_name().empty())
        return dim;
    return Dimension(m_dimension + dim.m_dimension);
}

Dimension Dimension::operator-(const Dimension& dim) const
{
    if (dim.m_dimension == 0 && dim.get_name().empty())
        return *this;
    return Dimension(m_dimension - dim.m_dimension);
}

Dimension Dimension::operator*(const Dimension& dim) const
{
    if (dim.m_dimension == 1 && dim.get_name().empty())
        return *this;
    else if (m_dimension == 1 && get_name().empty())
        return dim;
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
    std::string name;
    if (d1 == d2 && d1.get_name() == d2.get_name())
        name = d1.get_name();
    dst = {result, name};
    return true;
}

std::string broadcast_dimensions_name(const Dimension& d1, const Dimension& d2)
{
    std::string name;
    if (d1 == d2)
    {
        const auto& name_1 = d1.get_name();
        const auto& name_2 = d2.get_name();
        if (name_1 == name_2 || (!name_1.empty() && name_2.empty()))
            name = name_1;
        else if (name_1.empty() && !name_2.empty())
            name = name_2;
        return name;
    }

    const auto& one_dim = d1 == 1 ? d1 : (d2 == 1 ? d2 : -1);
    const auto& other_dim = d1 == 1 ? d2 : (d2 == 1 ? d1 : -1); // it is not equal to 1
    if (one_dim.is_dynamic())
        return "";
    return other_dim.get_name();
}

bool Dimension::broadcast_merge(Dimension& dst, const Dimension d1, const Dimension d2)
{
    if (d1.m_dimension.size() == 1 && d1.m_dimension.get_min_val() == 1)
    {
        dst =
            Dimension(d2.get_min_length(), d2.get_max_length(), broadcast_dimensions_name(d1, d2));
        return true;
    }
    if (d2.m_dimension.size() == 1 && d2.m_dimension.get_min_val() == 1)
    {
        dst =
            Dimension(d1.get_min_length(), d1.get_max_length(), broadcast_dimensions_name(d1, d2));
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
