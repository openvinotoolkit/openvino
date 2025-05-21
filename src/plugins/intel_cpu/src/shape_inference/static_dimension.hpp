// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <limits>
#include <ostream>
#include <stdexcept>

#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"

namespace ov::intel_cpu {

/// \brief Class representing a dimension, which must be static,
///        in a shape or shape-like object.
///
/// Provides similar API to the public Dimension class.
class StaticDimension {
public:
    using value_type = size_t;

    /// \brief Construct a static dimension.
    /// \param dimension Value of the dimension.
    StaticDimension(value_type dimension);

    /// \brief Construct a static dimension.
    /// \param ldimension Value of the dimension (must be equal to udimension)
    /// \param udimension Value of the dimension (must be equal to ldimension)
    StaticDimension(value_type ldimension, value_type udimension);

    /// \brief Construct a zero dimension
    StaticDimension() = default;

    StaticDimension(const Dimension&) {
        OPENVINO_THROW("[shape infer] Shoudn't convert from Dimension to StaticDimension.");
    }

    bool operator==(const StaticDimension& dimension) const;
    bool operator!=(const StaticDimension& dimension) const;

    explicit operator size_t() const {
        return m_dimension;
    }

    static constexpr bool is_static() {
        return true;
    }
    static constexpr bool is_dynamic() {
        return false;
    }

    [[nodiscard]] value_type get_length() const;
    [[nodiscard]] value_type get_min_length() const;
    [[nodiscard]] value_type get_max_length() const;

    [[nodiscard]] Interval& get_interval() const {
        static Interval dummy{};
        OPENVINO_THROW("[shape infer] Shoudn't call get_interval() in StaticDimension.");
        return dummy;
    }

    [[nodiscard]] bool same_scheme(const StaticDimension& dim) const;
    [[nodiscard]] bool compatible(const StaticDimension& d) const;
    static bool merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2);
    static bool broadcast_merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2);

    StaticDimension operator+(const StaticDimension& dim) const;
    StaticDimension operator-(const StaticDimension& dim) const;
    StaticDimension operator*(const StaticDimension& dim) const;
    StaticDimension operator&(const StaticDimension& dim) const;
    StaticDimension& operator+=(const StaticDimension& dim);
    StaticDimension& operator*=(const StaticDimension& dim);
    StaticDimension& operator&=(const StaticDimension& dim);
    StaticDimension operator/(const value_type divisor) const;
    StaticDimension& operator/=(const value_type divisor);

    /// \brief Swap of dimensions
    friend void swap(StaticDimension& a, StaticDimension& b) noexcept {
        using std::swap;
        swap(a.m_dimension, b.m_dimension);
    }

private:
    value_type m_dimension = 0;
};

std::ostream& operator<<(std::ostream& str, const StaticDimension& dimension);

}  // namespace ov::intel_cpu
