//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <stddef.h>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/dimension.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/rank.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace op
    {
        struct AutoBroadcastSpec;
    }

    /// \brief Class representing a shape that may be partially or totally dynamic.
    ///
    /// XXX: THIS CLASS IS EXPERIMENTAL AND THE ENTIRE DESIGN IS SUBJECT TO CHANGE.
    ///
    /// A PartialShape may have:
    ///
    /// \li Dynamic rank. (Informal notation: `?`)
    /// \li Static rank, but dynamic dimensions on some or all axes.
    ///     (Informal notation examples: `{1,2,?,4}`, `{?,?,?}`)
    /// \li Static rank, and static dimensions on all axes.
    ///     (Informal notation examples: `{1,2,3,4}`, `{6}`, `{}`)
    class NGRAPH_API PartialShape
    {
    public:
        /// \brief Constructs a shape with static rank from an initializer list of Dimension.
        /// \param init The Dimension values for the constructed shape.
        ///
        /// Examples:
        ///
        /// \code{.cpp}
        /// PartialShape s{2,3,4};                     // rank=3, all dimensions static
        /// PartialShape s{};                          // rank=0
        /// PartialShape s{2,Dimension::dynamic(),3};  // rank=3, dimension 1 dynamic
        /// \endcode
        PartialShape(std::initializer_list<Dimension> init)
            : PartialShape(true, init)
        {
        }

        /// \brief Constructs a PartialShape with static rank from a vector of Dimension.
        /// \param dimensions The Dimension values for the constructed shape.
        PartialShape(const std::vector<Dimension>& dimensions)
            : m_rank_is_static(true)
            , m_dimensions(dimensions)
        {
        }

        /// \brief Constructs a static PartialShape with zero rank (the shape of a scalar).
        PartialShape()
            : PartialShape(std::initializer_list<Dimension>{})
        {
        }

        /// \brief Constructs a static PartialShape from a Shape.
        /// \param shape The Shape to convert into PartialShape.
        PartialShape(const Shape& shape);

        /// \brief Check if this shape is static.
        /// \return `true` if this shape is static, else `false`.
        ///
        /// A shape is considered static if it has static rank, and all dimensions of the shape
        /// are static.
        bool is_static() const;

        /// \brief Check if this shape is dynamic.
        /// \return `false` if this shape is static, else `true`.
        ///
        /// A shape is considered static if it has static rank, and all dimensions of the shape
        /// are static.
        bool is_dynamic() const { return !is_static(); }
        /// \brief Get the rank of the shape.
        /// \return The rank of the shape. This will be Rank::dynamic() if the rank of
        ///         the shape is dynamic.
        Rank rank() const { return m_rank_is_static ? Rank(m_dimensions.size()) : Rank::dynamic(); }
        /// \brief Construct a PartialShape with the given rank and all dimensions (if any) dynamic.
        /// \return A PartialShape with the given rank, and all dimensions (if any) dynamic.
        static PartialShape dynamic(Rank r = Rank::dynamic());
        /// \brief Check whether this shape is compatible with the argument, i.e., whether it is
        ///        possible to merge them.
        /// \param s The shape to be checked for compatibility with this shape.
        /// \return `true` if this shape is compatible with `s`, else `false`.
        ///
        /// Two shapes are compatible if
        /// \li one or both of them has dynamic rank, or
        /// \li both shapes have dynamic and equal rank, and their dimensions are elementwise
        ///     compatible (see Dimension::compatible()).
        bool compatible(const PartialShape& s) const;

        /// \brief Check whether this shape represents the same scheme as the argument.
        /// \param s The shape whose scheme is being compared with this shape.
        /// \return `true` if this shape represents the same scheme as `s`, else `false`.
        ///
        /// Two shapes `s1` and `s2` represent the same scheme if
        /// \li they both have dynamic rank, or
        /// \li they both have static and equal rank `r`, and for every `i` from `0` to `r-1`,
        ///     `s1[i]` represents the same scheme as `s2[i]` (see Dimension::same_scheme()).
        bool same_scheme(const PartialShape& s) const;

        /// \brief Check whether this shape is a relaxation of the argument.
        /// \param s The shape which is being compared against this shape.
        /// \return `true` if this shape relaxes `s`, else `false`.
        ///
        /// Intuitively, a PartialShape `s1` is said to _relax_ `s2` (or _is a
        /// relaxation_ of `s2`) if it is "more permissive" than `s2`. In other
        /// words, `s1` is a relaxation of `s2` if anything you can form by
        /// plugging things into the dynamic dimensions of `s2` is also
        /// something you can form by plugging things into the dynamic
        /// dimensions of `s1`, but not necessarily the other way around.
        ///
        /// `s1.relaxes(s2)` is equivalent to `s2.refines(s1)`.
        ///
        /// Formally, PartialShape `s1` is said to _relax_ PartialShape `s2`
        /// if:
        /// \li For every `i` from `0` to `r-1`,
        ///      either `s1[i]` contains s2[i].
        bool relaxes(const PartialShape& s) const;

        /// \brief Check whether this shape is a refinement of the argument.
        /// \param s The shape which is being compared against this shape.
        /// \return `true` if this shape refines `s`, else `false`.
        ///
        /// Intuitively, a PartialShape `s1` is said to _relax_ `s2` (or _is a
        /// relaxation_ of `s2`) if it is "less permissive" than `s2`. In other
        /// words, `s1` is a relaxation of `s2` if anything you can form by
        /// plugging things into the dynamic dimensions of `s1` is also
        /// something you can form by plugging things into the dynamic
        /// dimensions of `s2`, but not necessarily the other way around.
        ///
        /// `s1.refines(s2)` is equivalent to `s2.relaxes(s1)`.
        ///
        /// Formally, PartialShape `s1` is said to _refine_ PartialShape `s2`
        /// if:
        /// \li `s2` has dynamic rank, or
        /// \li `s1` and `s2` both have static rank `r`, and for every `i` from `0` to `r-1`,
        ///      either `s2[i]` is dynamic, or `s1[i]` == `s2[i]`.
        bool refines(const PartialShape& s) const;

        /// \brief Checks that this shape's rank is compatible with `r`, and, if this shape's
        ///        rank is dynamic and `r` is static, updates this shape to have a rank of `r`
        ///        with dimensions all dynamic.
        /// \return `true` if this shape's rank is compatible with `r`, else `false`.
        bool merge_rank(Rank r);

        /// \brief Convert a static PartialShape to a Shape.
        /// \return A new Shape `s` where `s[i] = size_t((*this)[i])`.
        /// \throws std::invalid_argument If this PartialShape is dynamic.
        Shape to_shape() const;

        /// \brief Returns `true` if all static dimensions of the tensor are non-negative, else
        ///        `false`.
        bool all_non_negative() const;

        /// \brief Index operator for PartialShape.
        /// \param i The index of the dimension being selected.
        /// \return A reference to the `i`th Dimension of this shape.
        const Dimension& operator[](size_t i) const;
        /// \brief Index operator for PartialShape.
        /// \param i The index of the dimension being selected.
        /// \return A reference to the `i`th Dimension of this shape.
        Dimension& operator[](size_t i);
        /// \brief Returns a vector of the dimensions. This has no meaning if dynamic.
        explicit operator std::vector<Dimension>() const { return m_dimensions; }
        friend NGRAPH_API std::ostream& operator<<(std::ostream& str, const PartialShape& shape);
        friend PartialShape operator+(const PartialShape& s1, const PartialShape& s2);
        bool operator==(const PartialShape& partial_shape) const;
        bool operator!=(const PartialShape& partial_shape) const;
        /// Get the max bounding shape
        Shape get_max_shape() const;
        /// Get the min bounding shape
        Shape get_min_shape() const;
        /// Get the unique shape
        Shape get_shape() const;

        /// \brief Try to merge one shape into another.
        /// \param[in,out] dst The shape that `src` will be merged into.
        /// \param src The shape that will be merged into `dst`.
        /// \return `true` if merging succeeds, else `false`.
        ///
        /// Merges `src` into `dst`, returning `true` on success and `false` on failure. If
        /// `false` is returned, the effect on `dst` is unspecified.
        ///
        /// To merge two partial shapes `s1` and `s2` is to find the most permissive partial shape
        /// `s` that is no more permissive than `s1` or `s2`, if `s` exists. For example:
        ///
        /// \code
        ///        merge(?,?) -> ?
        ///        merge(?,{?,?}) -> {?,?}
        ///        merge({?,?},{?,?}) -> {?,?}
        ///        merge({1,2,3,4},?) -> {1,2,3,4}
        ///        merge({1,2},{1,?}) -> {1,2}
        ///        merge({1,2,?,?},{1,?,3,?}) -> {1,2,3,?}
        ///        merge({1,2,3},{1,2,3}) -> {1,2,3}
        ///
        ///        merge({1,?},{2,?}) fails [dimension 0 constraints are inconsistent]
        ///        merge({?,?},{?,?,?}) fails [ranks are inconsistent]
        /// \endcode
        ///
        /// This function (merge_into) performs the "merge" operation described above on `dst` and
        /// `src`, but overwrites `dst` with the result and returns `true` if merging is
        /// successful; if merging is unsuccessful, the function returns `false` and may make
        /// unspecified changes to `dst`.
        static bool merge_into(PartialShape& dst, const PartialShape& src);

        /// \brief Try to merge one shape into another along with implicit broadcasting
        static bool broadcast_merge_into(PartialShape& dst,
                                         const PartialShape& src,
                                         const op::AutoBroadcastSpec& autob);

    private:
        // Private constructor for PartialShape::dynamic().
        PartialShape(bool rank_is_static, std::vector<Dimension> dimensions)
            : m_rank_is_static(rank_is_static)
            , m_dimensions(dimensions)
        {
        }

        // True if the shape's rank is static.
        bool m_rank_is_static;

        // Shape dimensions. This has no meaning if m_rank_is_static is false.
        std::vector<Dimension> m_dimensions;
    };

    /// \brief Elementwise addition of two PartialShape objects.
    /// \param s1 Left operand for addition.
    /// \param s2 Right operand for addition.
    /// \return The result of elementwise adding `s1` to `s2` (see description).
    /// \throws std::invalid_argument If `s1` and `s2` have inconsistent ranks.
    ///
    /// \li If `s1` or `s2` has dynamic rank, returns PartialShape::dynamic().
    /// \li If `s1 and `s2` both have static rank, and their ranks are unequal, throws
    ///     std::invalid_argument.
    /// \li If `s1` and `s2` both have static rank, and their ranks are equal,
    ///     returns a new shape whose `i`th dimension is `s1[i] + s2[i]`.
    PartialShape operator+(const PartialShape& s1, const PartialShape& s2);

    /// \brief Inserts a human-readable representation of a PartialShape into an output stream.
    /// \param str The output stream targeted for insertion.
    /// \param shape The shape to be inserted into `str`.
    /// \return A reference to `str` after insertion.
    ///
    /// The output to the stream is in "informal" notation. In other words:
    ///
    /// \li If `shape` has dynamic rank, inserts the string `?`.
    /// \li If `shape` has static rank, inserts the string `{`, then inserts each dimension
    ///     of `shape` into the output stream separated by commas, then inserts `}`.
    ///
    /// Example:
    ///
    /// \code{.cpp}
    /// PartialShape s1{PartialShape::dynamic())};
    /// PartialShape s2{};
    /// PartialShape s3{1,Dimension::dynamic(),2,3};
    /// PartialShape s4{2,3,4};
    /// std::cout << s1 << std::endl
    ///           << s2 << std::endl
    ///           << s3 << std::endl
    ///           << s4 << std::endl;
    /// \endcode
    ///
    /// Output:
    ///
    /// \code
    /// ?
    /// {}
    /// {1,?,2,3}
    /// {2,3,4}
    /// \endcode
    NGRAPH_API
    std::ostream& operator<<(std::ostream& str, const PartialShape& shape);

    template <>
    class NGRAPH_API AttributeAdapter<PartialShape> : public ValueAccessor<std::vector<int64_t>>
    {
    public:
        AttributeAdapter(PartialShape& value)
            : m_ref(value)
        {
        }

        const std::vector<int64_t>& get() override;
        void set(const std::vector<int64_t>& value) override;
        static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<PartialShape>", 0};
        const DiscreteTypeInfo& get_type_info() const override { return type_info; }
        operator PartialShape&() { return m_ref; }
    protected:
        PartialShape& m_ref;
        std::vector<int64_t> m_buffer;
        bool m_buffer_valid{false};
    };
}
