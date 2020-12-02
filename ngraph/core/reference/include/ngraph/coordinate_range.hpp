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
            using ssize_t = typename std::make_signed<size_t>::type;

            namespace
            {
                template <typename C>
                bool has_zeros(const C& c)
                {
                    const auto is_zero = [](size_t x) { return x == 0; };
                    return std::any_of(c.begin(), c.end(), is_zero);
                }

            } // namespace
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

                ssize_t last_dim_size() const noexcept { return m_upper.back() - m_lower.back(); }
            };

            template <typename Range>
            struct RangeBase
            {
                using Iterator = RangeIterator<Range>;

                Iterator begin() { return Iterator(static_cast<Range*>(this)); }
                Iterator end() { return Iterator(nullptr); }
            };

            struct Range
            {
                const ssize_t begin;
                const ssize_t end;
                const ssize_t step;
                bool is_in(ssize_t i) const
                {
                    return step > 0 ? (begin <= i && i < end) : (end < i && i <= begin);
                }
            };

            class SliceRange : public RangeBase<SliceRange>
            {
            public:
                using value_type = Range;
                SliceRange(const Shape& source_shape,
                           const Coordinate& source_start_corner,
                           const Coordinate& source_end_corner,
                           const Strides& strides);

                ssize_t copy_range_first_index() const;

                value_type get_value() const
                {
                    return Range{m_index,
                                 m_index + m_bounds.last_dim_size(),
                                 static_cast<ssize_t>(m_source_strides.back())};
                }

                bool increment();

                bool is_valid() const noexcept { return !has_zeros(m_source_shape); }
                Coordinate coodinate() const { return m_coordinate; }
            private:
                const Shape m_source_shape;
                const CoordinateBounds m_bounds;
                const Strides m_source_strides;
                const std::vector<ssize_t> m_memory_strides;
                Coordinate m_coordinate;
                ssize_t m_index{0};
            };

            inline SliceRange slice(const Shape& source_shape,
                                    const Coordinate& source_start_corner,
                                    const Coordinate& source_end_corner,
                                    const Strides& strides)
            {
                return SliceRange{source_shape, source_start_corner, source_end_corner, strides};
            }
            inline SliceRange slice(const Shape& source_shape,
                                    const Coordinate& source_start_corner,
                                    const Coordinate& source_end_corner)
            {
                return slice(source_shape,
                             source_start_corner,
                             source_end_corner,
                             Strides(source_shape.size(), 1));
            }

            class ReverseRange : public RangeBase<ReverseRange>
            {
            public:
                using value_type = Range;
                ReverseRange(const Shape& source_shape, const AxisSet& reversed_axis);

                value_type get_value() const;

                bool increment();

                bool is_valid() const noexcept { return !has_zeros(m_source_shape); }
            private:
                ssize_t last_dim_size() const noexcept;

                const Shape m_source_shape;
                const std::vector<ssize_t> m_memory_strides;
                const std::vector<signed char> m_axis_directions;
                Coordinate m_coordinate;
                ssize_t m_index{0};
            };

            inline ReverseRange reverse(const Shape& source_shape, const AxisSet& reversed_axis)
            {
                return ReverseRange(source_shape, reversed_axis);
            }

        } // namespace impl
        using impl::reverse;
        using impl::slice;
    } // namespace coordinates
} // namespace ngraph
