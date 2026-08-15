// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local dimension layer. Minimal replacement for openvino/core/dimension.hpp.

#pragma once

#include <cstdint>
#include <limits>

namespace ov {

class Dimension {
public:
    using value_type = int64_t;

    static constexpr value_type DYNAMIC_DIMENSION = -1;

    Dimension() : m_dimension(DYNAMIC_DIMENSION) {}
    Dimension(value_type dim) : m_dimension(dim) {}

    value_type get_length() const { return m_dimension; }
    value_type get_min_length() const { return m_dimension; }
    value_type get_max_length() const { return m_dimension; }

    bool is_dynamic() const { return m_dimension < 0; }
    bool is_static() const { return m_dimension >= 0; }

    operator value_type() const { return m_dimension; }

    friend bool operator==(const Dimension& a, const Dimension& b) { return a.m_dimension == b.m_dimension; }
    friend bool operator!=(const Dimension& a, const Dimension& b) { return a.m_dimension != b.m_dimension; }
    friend bool operator<(const Dimension& a, const Dimension& b) { return a.m_dimension < b.m_dimension; }

private:
    value_type m_dimension = DYNAMIC_DIMENSION;
};

}  // namespace ov
