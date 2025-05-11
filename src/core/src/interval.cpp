// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/interval.hpp"

using namespace ov;

namespace {
Interval::value_type clip(Interval::value_type value) {
    return std::max(Interval::value_type(0), std::min(Interval::s_max, value));
}

Interval::value_type clip_times(Interval::value_type a, Interval::value_type b) {
    if (a == 0 || b == 0) {
        return 0;
    } else if (a == Interval::s_max || b == Interval::s_max || a > Interval::s_max / b) {
        return Interval::s_max;
    } else {
        return a * b;
    }
}
Interval::value_type clip_add(Interval::value_type a, Interval::value_type b) {
    if (a == Interval::s_max || b == Interval::s_max) {
        return Interval::s_max;
    }

    // check overflow without undefined behavior: a + b <= max
    const static auto max = std::numeric_limits<Interval::value_type>::max();
    if (b > (max - a)) {
        return Interval::s_max;
    }

    return a + b;
}
Interval::value_type clip_minus(Interval::value_type a, Interval::value_type b) {
    if (a <= b) {
        return 0;
    }
    if (a == Interval::s_max) {
        return Interval::s_max;
    }
    return a - b;
}
}  // namespace

void Interval::canonicalize() {
    if (m_max_val < m_min_val) {
        m_min_val = s_max;
        m_max_val = s_max;
    } else {
        m_min_val = clip(m_min_val);
        m_max_val = clip(m_max_val);
    }
}

Interval::Interval(value_type min_val, value_type max_val) : m_min_val(min_val), m_max_val(max_val) {
    canonicalize();
}

Interval::Interval(value_type val) {
    m_min_val = clip(val);
    m_max_val = m_min_val;
}

bool Interval::operator==(const Interval& interval) const {
    return m_min_val == interval.m_min_val && m_max_val == interval.m_max_val;
}

bool Interval::operator!=(const Interval& interval) const {
    return !(*this == interval);
}

Interval Interval::operator+(const Interval& interval) const {
    if (empty() || interval.empty()) {
        return Interval(s_max);
    }
    return Interval(clip_add(m_min_val, interval.m_min_val), clip_add(m_max_val, interval.m_max_val));
}

Interval& Interval::operator+=(const Interval& interval) {
    return *this = *this + interval;
}

Interval Interval::operator-(const Interval& interval) const {
    if (empty() || interval.empty()) {
        return Interval(s_max);
    }
    return Interval(clip_minus(m_min_val, interval.m_max_val), clip_minus(m_max_val, interval.m_min_val));
}

Interval& Interval::operator-=(const Interval& interval) {
    return *this = *this - interval;
}

Interval Interval::operator*(const Interval& interval) const {
    if (empty()) {
        return *this;
    }
    if (interval.empty()) {
        return interval;
    }
    return Interval(clip_times(m_min_val, interval.m_min_val), clip_times(m_max_val, interval.m_max_val));
}

Interval& Interval::operator*=(const Interval& interval) {
    return *this = *this * interval;
}

Interval Interval::operator&(const Interval& interval) const {
    return Interval(std::max(m_min_val, interval.m_min_val), std::min(m_max_val, interval.m_max_val));
}

Interval& Interval::operator&=(const Interval& interval) {
    return *this = *this & interval;
}

bool Interval::contains(const Interval& interval) const {
    return contains(interval.m_min_val) && contains(interval.m_max_val);
}

constexpr Interval::value_type Interval::s_max;

std::ostream& ov::operator<<(std::ostream& str, const Interval& interval) {
    str << "Interval(" << interval.get_min_val() << ", ";
    auto max_val = interval.get_max_val();
    if (max_val == Interval::s_max) {
        str << "...";
    } else {
        str << max_val;
    }
    return str << ")";
}
