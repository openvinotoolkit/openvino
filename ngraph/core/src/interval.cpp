// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/interval.hpp"

using namespace ngraph;

void Interval::canonicalize()
{
    if (m_max_val < m_min_val)
    {
        m_min_val = s_max;
        m_max_val = s_max;
    }
    else
    {
        m_min_val = clip(m_min_val);
        m_max_val = clip(m_max_val);
    }
}

Interval::Interval(value_type min_val, value_type max_val)
    : m_min_val(min_val)
    , m_max_val(max_val)
{
    canonicalize();
}

Interval::Interval(value_type val)
    : Interval(val, val)
{
}

Interval::size_type Interval::size() const
{
    if (m_max_val == s_max)
    {
        return m_min_val == s_max ? 0 : s_max;
    }
    return m_max_val - m_min_val + 1;
}

bool Interval::empty() const
{
    return m_min_val == s_max;
}

bool Interval::operator==(const Interval& interval) const
{
    return m_min_val == interval.m_min_val && m_max_val == interval.m_max_val;
}

bool Interval::operator!=(const Interval& interval) const
{
    return !(*this == interval);
}

Interval Interval::operator+(const Interval& interval) const
{
    if (empty() || interval.empty())
    {
        return Interval(s_max);
    }
    return Interval(clip_add(m_min_val, interval.m_min_val),
                    clip_add(m_max_val, interval.m_max_val));
}

Interval& Interval::operator+=(const Interval& interval)
{
    return *this = *this + interval;
}

Interval Interval::operator-(const Interval& interval) const
{
    if (empty() || interval.empty())
    {
        return Interval(s_max);
    }
    return Interval(clip_minus(m_min_val, interval.m_max_val),
                    clip_minus(m_max_val, interval.m_min_val));
}

Interval& Interval::operator-=(const Interval& interval)
{
    return *this = *this - interval;
}

Interval Interval::operator*(const Interval& interval) const
{
    if (empty())
    {
        return *this;
    }
    if (interval.empty())
    {
        return interval;
    }
    return Interval(clip_times(m_min_val, interval.m_min_val),
                    clip_times(m_max_val, interval.m_max_val));
}

Interval& Interval::operator*=(const Interval& interval)
{
    return *this = *this * interval;
}

Interval Interval::operator&(const Interval& interval) const
{
    return Interval(std::max(m_min_val, interval.m_min_val),
                    std::min(m_max_val, interval.m_max_val));
}

Interval& Interval::operator&=(const Interval& interval)
{
    return *this = *this & interval;
}

bool Interval::contains(value_type value) const
{
    return m_min_val <= value && value <= m_max_val;
}

bool Interval::contains(const Interval& interval) const
{
    return contains(interval.m_min_val) && contains(interval.m_max_val);
}

Interval::value_type Interval::clip(value_type value)
{
    return std::max(value_type(0), std::min(s_max, value));
}

Interval::value_type Interval::clip_add(value_type a, value_type b)
{
    return (a == s_max || b == s_max) ? s_max : a + b;
}

Interval::value_type Interval::clip_minus(value_type a, value_type b)
{
    if (a <= b)
    {
        return 0;
    }
    if (a == s_max)
    {
        return s_max;
    }
    return a - b;
}

Interval::value_type Interval::clip_times(value_type a, value_type b)
{
    if (a == 0 || b == 0)
    {
        return 0;
    }
    else if (a == s_max || b == s_max)
    {
        return s_max;
    }
    else
    {
        return a * b;
    }
}

constexpr Interval::value_type Interval::s_max;

namespace ngraph
{
    std::ostream& operator<<(std::ostream& str, const Interval& interval)
    {
        str << "Interval(" << interval.get_min_val() << ", ";
        auto max_val = interval.get_max_val();
        if (max_val == Interval::s_max)
        {
            str << "...";
        }
        else
        {
            str << max_val;
        }
        return str << ")";
    }
} // namespace ngraph
