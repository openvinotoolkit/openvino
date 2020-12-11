//*****************************************************************************
// Copyright 2020 Intel Corporation
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

#include <algorithm>
#include <iterator>

#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace coordinates
    {
        namespace impl
        {
            namespace
            {
                template <typename C>
                bool has_zeros(const C& c)
                {
                    const auto is_zero = [](size_t x) { return x == 0; };
                    return std::any_of(c.begin(), c.end(), is_zero);
                }

            } // namespace

            /// \brief Class which allow to iterate over a different ranges part by part
            ///
            template <typename Range>
            class RangeIterator
            {
            public:
                using value_type = typename Range::value_type;
                using reference = typename Range::value_type;
                using iterator_category = std::input_iterator_tag;
                using difference_type = void;

                RangeIterator(Range* r)
                    : m_r{r}
                {
                    if (m_r && !m_r->is_valid())
                    {
                        m_r = nullptr;
                    }
                }

                value_type operator*() const { return m_r->get_value(); }
                RangeIterator& operator++()
                {
                    if (m_r && !m_r->increment())
                    {
                        m_r = nullptr;
                    }
                    return *this;
                }

                RangeIterator operator++(int) = delete;

                friend bool operator==(const RangeIterator& lhs, const RangeIterator& rhs)
                {
                    return lhs.m_r == rhs.m_r;
                }
                friend bool operator!=(const RangeIterator& lhs, const RangeIterator& rhs)
                {
                    return !(lhs == rhs);
                }

            private:
                Range* m_r;
            };

            /// \brief Describe slice range
            ///
            struct CoordinateBounds
            {
                CoordinateBounds(const Coordinate& lower, const Coordinate& upper)
                    : m_lower{lower}
                    , m_upper{upper}
                {
                    if (m_lower.size() != m_upper.size())
                    {
                        throw std::domain_error{"different Coordinates bonds sizes"};
                    }
                }
                Coordinate m_lower;
                Coordinate m_upper;

                size_t last_dim_size() const noexcept { return m_upper.back() - m_lower.back(); }
            };

            /// \brief helper for iterator creation which allow to stay DRY
            ///
            template <typename Range>
            struct RangeBase
            {
                using Iterator = RangeIterator<Range>;

                Iterator begin() { return Iterator(static_cast<Range*>(this)); }
                Iterator end() { return Iterator(nullptr); }
                friend Iterator begin(Range& r) { return r.begin(); }
                friend Iterator end(Range& r) { return r.end(); }
            };

            /// \brief Information how index in _Range_ should be change
            ///
            enum class Direction
            {
                forward,
                reverse,
            };

            /// \brief Range contain information which part of memory should be copied
            ///
            struct Range
            {
                constexpr Range(size_t begin_index = 0,
                                size_t element_number = 0,
                                size_t step = 1,
                                Direction direction = Direction::forward)
                    : begin_index{begin_index}
                    , element_number{element_number}
                    , step{step}
                    , direction{direction}
                {
                }
                size_t begin_index;
                size_t element_number;
                size_t step;
                Direction direction;

                static constexpr Range make_empyt() { return Range{}; }
            };

            /// \brief Class allows to iterate over sliced Tensor part by part.
            ///
            /// To create SliceRange use _slice_ function.
            class SliceRange : public RangeBase<SliceRange>
            {
            public:
                using value_type = Range;
                SliceRange(const Shape& source_shape,
                           const Coordinate& source_start_corner,
                           const Coordinate& source_end_corner,
                           const Strides& strides);

                value_type get_value() const;

                bool increment();

                bool is_valid() const noexcept { return !has_zeros(m_source_shape); }

            private:
                const Shape m_source_shape;
                const CoordinateBounds m_bounds;
                const Strides m_source_strides;
                const std::vector<size_t> m_memory_strides;
                Coordinate m_coordinate;
                size_t m_index{0};
            };

            /// \brief Create SliceRange which might be used in range-base for loop
            ///
            inline SliceRange slice(const Shape& source_shape,
                                    const Coordinate& source_start_corner,
                                    const Coordinate& source_end_corner,
                                    const Strides& strides)
            {
                return SliceRange{source_shape, source_start_corner, source_end_corner, strides};
            }

            /// \brief Create SliceRange which might be used in range-base for loop
            ///
            inline SliceRange slice(const Shape& source_shape,
                                    const Coordinate& source_start_corner,
                                    const Coordinate& source_end_corner)
            {
                return slice(source_shape,
                             source_start_corner,
                             source_end_corner,
                             Strides(source_shape.size(), 1));
            }

            /// \brief Class allows to iterate over Tensor with reverted axies part by part.
            ///
            /// To create ReverseRange use _reverse_ function.
            ///
            class ReverseRange : public RangeBase<ReverseRange>
            {
            public:
                using value_type = Range;
                ReverseRange(const Shape& source_shape, const AxisSet& reversed_axis);

                value_type get_value() const;

                bool increment();

                bool is_valid() const noexcept { return !has_zeros(m_source_shape); }

            private:
                const Shape m_source_shape;
                const std::vector<size_t> m_memory_strides;
                const std::vector<Direction> m_axis_directions;
                Coordinate m_coordinate;
                size_t m_index{0};
            };

            inline ReverseRange reverse(const Shape& source_shape, const AxisSet& reversed_axis)
            {
                return ReverseRange(source_shape, reversed_axis);
            }

            template <typename TheRange>
            class IndexRagne : public RangeBase<IndexRagne<TheRange>>
            {
            public:
                IndexRagne(TheRange r)
                    : m_r{std::move(r)}
                    , m_current{m_r.get_value(), 0}
                {
                }
                using value_type = size_t;

                value_type get_value() const
                {
                    const auto offset_from_index = m_current.element * m_current.range.step;
                    if (m_current.range.direction == Direction::forward)
                    {
                        return m_current.range.begin_index + offset_from_index;
                    }
                    return m_current.range.begin_index - offset_from_index;
                }

                bool increment()
                {
                    ++m_current.element;

                    if (m_current.element >= m_current.range.element_number)
                    {
                        m_current.element = 0;
                        const bool there_is_more_indices = m_r.increment();
                        if (!there_is_more_indices)
                        {
                            return false;
                        }
                        m_current.range = m_r.get_value();
                    }
                    return true;
                }

                bool is_valid() const noexcept { return m_r.is_valid(); }

            private:
                TheRange m_r;
                struct Current
                {
                    Current(Range r = Range::make_empyt(), size_t element = 0)
                        : range{std::move(r)}
                        , element{element}
                    {
                    }
                    Range range{Range::make_empyt()};
                    size_t element{0};
                };
                Current m_current;
            };

            template <typename TheRange>
            IndexRagne<TheRange> index_range(TheRange r)
            {
                return IndexRagne<TheRange>(std::move(r));
            }
        } // namespace impl

        using impl::Direction;
        using impl::index_range;
        using impl::reverse;
        using impl::slice;
    } // namespace coordinates
} // namespace ngraph
