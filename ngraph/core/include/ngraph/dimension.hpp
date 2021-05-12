// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <stddef.h>
#include <stdexcept>

#include "ngraph/deprecated.hpp"
#include "ngraph/interval.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph
{
    /// \brief Class representing a dimension, which may be dynamic (undetermined until runtime),
    ///        in a shape or shape-like object.
    ///
    /// Static dimensions may be implicitly converted from value_type. A dynamic dimension is
    /// constructed with Dimension() or Dimension::dynamic().
    class NGRAPH_API Dimension
    {
    public:
        using value_type = int64_t;

        /// \brief Construct a static dimension.
        /// \param dimension Value of the dimension.
        Dimension(value_type dimension);

        /// \brief Construct a dynamic dimension with bounded range
        /// \param min_dimension The lower inclusive limit for the dimension
        /// \param mas_dimension The upper inclusive limit for the dimension
        Dimension(value_type min_dimension, value_type max_dimension);

        /// \brief Construct a dynamic dimension with range [0, ...]
        Dimension() = default;

        bool operator==(const Dimension& dimension) const
        {
            return m_dimension == dimension.m_dimension;
        }
        bool operator!=(const Dimension& dimension) const
        {
            return m_dimension != dimension.m_dimension;
        }
        /// \brief Check whether this dimension is static.
        /// \return `true` if the dimension is static, else `false`.
        bool is_static() const { return m_dimension.size() == 1; }
        /// \brief Check whether this dimension is dynamic.
        /// \return `false` if the dimension is static, else `true`.
        bool is_dynamic() const { return m_dimension.size() != 1; }
        /// \brief Convert this dimension to `value_type`. This dimension must be static and
        ///        non-negative.
        /// \throws std::invalid_argument If this dimension is dynamic or negative.
        value_type get_length() const;

        value_type get_min_length() const;
        value_type get_max_length() const;

        /// \brief Return the interval of valid lengths
        const Interval& get_interval() const { return m_dimension; }
        Interval& get_interval() { return m_dimension; }
        /// \brief Check whether this dimension represents the same scheme as the argument (both
        ///        dynamic, or equal).
        /// \param dim The other dimension to compare this dimension to.
        /// \return `true` if this dimension and `dim` are both dynamic, or if they are both
        ///         static and equal; otherwise, `false`.
        bool same_scheme(const Dimension& dim) const;
        /// \brief Try to merge two Dimension objects together.
        /// \param[out] dst Reference to write the merged Dimension into.
        /// \param d1 First dimension to merge.
        /// \param d2 Second dimension to merge.
        /// \return `true` if merging succeeds, else `false`.
        ///
        /// \li If `d1` is dynamic, writes `d2` to `dst` and returns `true`.
        /// \li If `d2` is dynamic, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are static and equal, writes `d1` to `dst` and returns `true`.
        /// \li If `d1` and `d2` are both static and unequal, leaves `dst` unchanged and
        ///     returns `false`.
        static bool merge(Dimension& dst, const Dimension d1, const Dimension d2);

        /// \brief Try to merge two Dimension objects together with implicit broadcasting
        ///        of unit-sized dimension to non unit-sized dimension
        static bool broadcast_merge(Dimension& dst, const Dimension d1, const Dimension d2);

        /// \brief Check whether this dimension is capable of being merged with the argument
        ///        dimension.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension is compatible with `d`, else `false`.
        ///
        /// Two dimensions are considered compatible if it is possible to merge them. (See
        /// Dimension::merge.)
        bool compatible(const Dimension& d) const;

        /// \brief Check whether this dimension is a relaxation of the argument.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension relaxes `d`, else `false`.
        ///
        /// A dimension `d1` _relaxes_ (or _is a relaxation of_) `d2` if `d1` and `d2` are static
        /// and equal, or `d1` is dynamic.
        ///
        /// `d1.relaxes(d2)` is equivalent to `d2.refines(d1)`.
        bool relaxes(const Dimension& d) const;

        /// \brief Check whether this dimension is a refinement of the argument.
        /// \param d The dimension to compare this dimension with.
        /// \return `true` if this dimension relaxes `d`, else `false`.
        ///
        /// A dimension `d2` _refines_ (or _is a refinement of_) `d1` if `d1` and `d2` are static
        /// and equal, or `d2` is dynamic.
        ///
        /// `d1.refines(d2)` is equivalent to `d2.relaxes(d1)`.
        bool refines(const Dimension& d) const;

        /// \brief Create a dynamic dimension.
        /// \return A dynamic dimension.
        static Dimension dynamic() { return Dimension(); }
        /// \brief Addition operator for Dimension.
        /// \param dim Right operand for addition.
        /// \return Smallest interval dimension enclosing inputs
        Dimension operator+(const Dimension& dim) const;

        /// \brief Subtraction operator for Dimension.
        /// \param dim Right operand for subtraction.
        /// \return Smallest interval dimension enclosing inputs
        Dimension operator-(const Dimension& dim) const;

        /// \brief Multiplication operator for Dimension.
        /// \param dim Right operand for multiplicaiton.
        /// \return Smallest interval containing all "produces" which are 0 if either of `this` or
        /// `dim` has length `0`, else unbounded if either is unbounded, else product of lengths.
        Dimension operator*(const Dimension& dim) const;

        /// \brief Add-into operator for Dimension.
        /// \param dim Right operand for addition.
        /// \return A reference to `*this`, after updating `*this` to the value `*this + dim`.
        Dimension& operator+=(const Dimension& dim) { return (*this = *this + dim); }
        /// \brief Multiply-into operator for Dimension.
        /// \param dim Right operand for multiplication.
        /// \return A reference to `*this`, after updating `*this` to the value `*this * dim`.
        Dimension& operator*=(const Dimension& dim) { return (*this = *this * dim); }
        /// \brief Intersection of dimensions
        Dimension operator&(const Dimension& dim) const;
        /// \brief Intersection of dimensions
        Dimension& operator&=(const Dimension& dim);

    private:
        Dimension(const Interval& interval)
            : m_dimension(interval)
        {
        }

        // The actual numerical value of the dimension.
        Interval m_dimension{};
    };

    /// \brief Insert a human-readable representation of a dimension into an output stream.
    /// \param str The output stream targeted for insertion.
    /// \param dimension The dimension to be inserted into `str`.
    /// \return A reference to `str` after insertion.
    ///
    /// Inserts the string `?` if `dimension` is dynamic; else inserts `dimension.get_length()`.
    NGRAPH_API
    std::ostream& operator<<(std::ostream& str, const Dimension& dimension);
} // namespace ngraph
