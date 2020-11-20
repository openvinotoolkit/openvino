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
        template <typename Range>
        class RangeIterator
        {
            template <typename C>
            bool has_zeros(const C& c)
            {
                const auto is_zero = [](size_t x) { return x == 0; };
                return std::any_of(c.begin(), c.end(), is_zero);
            }

        public:
            using value_type = Range;
            using reference = Range&;
            using iterator_category = std::input_iterator_tag;
            using pointer = Range*;
            using difference_type = void;

            RangeIterator(Range* r)
                : m_r{r}
            {
                if (m_r && has_zeros(m_r->source_shape()))
                {
                    m_r = nullptr;
                }
            }

            const Range& operator*() const { return *m_r; }
            const Range* operator->() const { return m_r; }
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

        class SliceRange
        {
        public:
            SliceRange(const Shape& source_shape,
                       const Coordinate& source_start_corner,
                       const Coordinate& source_end_corner,
                       const Strides& strides);

            size_t index() const;

            bool increment();

            const Shape& source_shape() const { return m_source_shape; }
            const Coordinate& coodinate() const { return m_coordinate; }
            using Iterator = RangeIterator<SliceRange>;
            Iterator begin() { return Iterator(this); }
            Iterator end() { return Iterator(nullptr); }
        private:
            struct CoordinateBounds
            {
                Coordinate lower;
                Coordinate upper;
            };
            const Shape m_source_shape;
            const CoordinateBounds m_bounds;
            const Strides m_source_strides;
            Coordinate m_coordinate;
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

        class ReverseRange
        {
        public:
            ReverseRange(const Shape& source_shape, const AxisSet& reversed_axes);

            size_t index() const;

            bool increment();

            const Shape& source_shape() const { return m_source_shape; }
            const Coordinate& coodinate() const
            {
                do_reverse();
                return m_reversed_coordinate;
            }

            using Iterator = RangeIterator<ReverseRange>;
            Iterator begin() { return Iterator(this); }
            Iterator end() { return Iterator(nullptr); }
        private:
            void do_reverse() const;
            const Shape m_source_shape;
            const AxisSet m_reversed_axes;
            Coordinate m_coordinate;
            mutable Coordinate m_reversed_coordinate;
        };

        inline ReverseRange reverse(const Shape& source_shape, const AxisSet& reversed_axes)
        {
            return ReverseRange(source_shape, reversed_axes);
        }

    } // namespace coordinates
} // namespace ngraph
