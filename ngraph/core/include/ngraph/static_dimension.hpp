// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stddef.h>

#include <limits>
#include <stdexcept>

#include "ngraph/check.hpp"
#include "ngraph/deprecated.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph {
/// \brief Class representing a dimension in a shape or shape-like object.
///
/// Static dimensions may be implicitly converted from value_type.
class NGRAPH_API StaticDimension {
public:
    using value_type = size_t;

    /// \brief Construct a static dimension.
    /// \param dimension Value of the dimension.
    StaticDimension(value_type dimension) : m_dimension(dimension){};
    StaticDimension(Dimension dimension) : m_dimension(dimension.get_max_length()) {
        NGRAPH_CHECK(dimension.is_static());
    };

    /// \brief Construct a static zero dimension
    StaticDimension() = default;

    bool operator==(const StaticDimension& dimension) const {
        return m_dimension == dimension.m_dimension;
    }
    bool operator!=(const StaticDimension& dimension) const {
        return m_dimension != dimension.m_dimension;
    }
    /// \brief Check whether this dimension is static.
    /// \return `true` if the dimension is static, else `false`.
    static bool is_static() {
        return true;
    }
    /// \brief Check whether this dimension is dynamic.
    /// \return `false` if the dimension is static, else `true`.
    static bool is_dynamic() {
        return false;
    }
    /// \brief Convert this dimension to `value_type`. This dimension must be static and
    ///        non-negative.
    /// \throws std::invalid_argument If this dimension is dynamic or negative.
    value_type get_length() const {
        return m_dimension;
    };

    /// \brief Check whether this dimension represents the same scheme as the argument (equal).
    /// \param dim The other dimension to compare this dimension to.
    /// \return `true` if this dimension and `dim` are equal; otherwise, `false`.
    bool same_scheme(const StaticDimension& dim) const {
        return *this == dim;
    };

    /// \brief Try to merge two StaticDimension objects together.
    /// \param[out] dst Reference to write the merged StaticDimension into.
    /// \param d1 First dimension to merge.
    /// \param d2 Second dimension to merge.
    /// \return `true` if merging succeeds, else `false`.
    ///
    /// \li If `d1` and `d2` are equal, writes `d1` to `dst` and returns `true`.
    /// \li If `d1` and `d2` are unequal, leaves `dst` unchanged and returns `false`.
    static bool merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2) {
        bool equal = d1 == d2;
        if (equal)
            dst = d1;
        return equal;
    }

    /// \brief Try to merge two StaticDimension objects together with implicit broadcasting
    ///        of unit-sized dimension to non unit-sized dimension
    static bool broadcast_merge(StaticDimension& dst, const StaticDimension& d1, const StaticDimension& d2) {
        bool status = true;
        if (d1 == 1)
            dst = d2;
        else if (d2 == 1)
            dst = d1;
        else
            status = merge(dst, d1, d2);
        return status;
    }

    bool compatible(const StaticDimension& d) const {
        return *this == d;
    };
    bool relaxes(const StaticDimension& d) const {
        return *this == d;
    };
    bool refines(const StaticDimension& d) const {
        return *this == d;
    };

    StaticDimension operator+(const StaticDimension& dim) const {
        return m_dimension + dim.m_dimension;
    }
    StaticDimension operator-(const StaticDimension& dim) const {
        return m_dimension - dim.m_dimension;
    }
    StaticDimension operator*(const StaticDimension& dim) const {
        return m_dimension * dim.m_dimension;
    }
    StaticDimension operator/(const StaticDimension& dim) const {
        return m_dimension / dim.m_dimension;
    }
    StaticDimension operator%(const StaticDimension& dim) const {
        return m_dimension % dim.m_dimension;
    }

    bool operator<(const StaticDimension& dim) const {
        return m_dimension < dim.m_dimension;
    }
    bool operator<=(const StaticDimension& dim) const {
        return m_dimension <= dim.m_dimension;
    }
    bool operator>(const StaticDimension& dim) const {
        return m_dimension > dim.m_dimension;
    }
    bool operator>=(const StaticDimension& dim) const {
        return m_dimension >= dim.m_dimension;
    }

    StaticDimension& operator+=(const StaticDimension& dim) {
        return (*this = *this + dim);
    }
    StaticDimension& operator*=(const StaticDimension& dim) {
        return (*this = *this * dim);
    }
    StaticDimension& operator/=(const StaticDimension& dim) {
        return (*this = *this / dim);
    }

    StaticDimension operator&(const StaticDimension& dim) const {
        NGRAPH_CHECK(*this == dim);
        return dim;
    }
    StaticDimension& operator&=(const StaticDimension& dim) {
        return (*this = *this & dim);
    }

    friend NGRAPH_API std::ostream& operator<<(std::ostream& str, const StaticDimension& shape);

private:
    value_type m_dimension;
};

/// \brief Insert a human-readable representation of a dimension into an output stream.
/// \param str The output stream targeted for insertion.
/// \param dimension The dimension to be inserted into `str`.
/// \return A reference to `str` after insertion.
NGRAPH_API
std::ostream& operator<<(std::ostream& str, const StaticDimension& dimension);
}  // namespace ngraph
