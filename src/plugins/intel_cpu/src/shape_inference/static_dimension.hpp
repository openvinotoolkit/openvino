// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <ostream>

#include "openvino/core/dimension.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/interval.hpp"

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
    ///
    /// \brief Construct a static dimension.
    /// \param dimension Value of the dimension.
    ///
    /// \note Implicit conversion is intentionally allowed (explicit constructor disabled)
    ///       because StaticDimension is used quite intensively throughout the codebase
    ///       and requiring explicit conversions would make the code more verbose.
    StaticDimension(value_type dimension);  // NOLINT(google-explicit-constructor)

    /// \brief Construct a static dimension.
    /// \param ldimension Value of the dimension (must be equal to udimension)
    /// \param udimension Value of the dimension (must be equal to ldimension)
    StaticDimension(value_type ldimension, value_type udimension);

    /// \brief Construct a zero dimension
    StaticDimension() = default;

    StaticDimension(const Dimension& dim);  // NOLINT(google-explicit-constructor)

    bool operator==(const StaticDimension& dimension) const;
    bool operator!=(const StaticDimension& dimension) const;
    bool operator!=(value_type val) const;
    bool operator!=(int val) const;

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

    [[nodiscard]] static Interval& get_interval() {
        static Interval dummy{};
        OPENVINO_THROW("[shape infer] Shoudn't call get_interval() in StaticDimension.");
        return dummy;
    }

    [[nodiscard]] bool same_scheme(const StaticDimension& dim) const;
    [[nodiscard]] bool compatible(const StaticDimension& d) const;
    [[nodiscard]] bool compatible(value_type d) const;
    static bool merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2);
    static bool broadcast_merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2);

    StaticDimension operator+(const StaticDimension& dim) const;
    StaticDimension operator-(const StaticDimension& dim) const;
    StaticDimension operator*(const StaticDimension& dim) const;
    StaticDimension operator&(const StaticDimension& dim) const;
    StaticDimension operator+(value_type val) const;
    StaticDimension operator-(value_type val) const;
    StaticDimension operator*(value_type val) const;
    StaticDimension operator*(const ov::Dimension& dim) const;
    bool operator!=(const ov::Dimension& dim) const;
    bool operator==(value_type val) const;
    bool operator==(int val) const;
    bool operator==(const ov::Dimension& dim) const;
    StaticDimension& operator+=(const StaticDimension& dim);
    StaticDimension& operator*=(const StaticDimension& dim);
    StaticDimension& operator&=(const StaticDimension& dim);
    StaticDimension& operator=(const ov::Dimension& dim);
    StaticDimension& operator=(value_type val);
    StaticDimension operator/(value_type divisor) const;
    StaticDimension& operator/=(value_type divisor);

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
