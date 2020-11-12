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

#include <iterator>

#include "ngraph/coordinate.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    namespace coordinates
    {
        class SliceRange
        {
        public:
            SliceRange(const Shape& source_shape,
                       const Coordinate& source_start_corner,
                       const Coordinate& source_end_corner,
                       const Strides& strides)
                : m_source_shape{source_shape}
                , m_bounds{source_start_corner, source_end_corner}
                , m_strides{strides}
                , m_coordinate{source_start_corner}
            {
            }

            size_t index() const;

            class Iterator
            {
            public:
                using value_type = SliceRange;
                using reference = SliceRange&;
                using iterator_category = std::input_iterator_tag;
                using pointer = SliceRange*;
                using difference_type = void;

                Iterator(SliceRange* r)
                    : m_r{r}
                {
                }

                const SliceRange& operator*() const { return *m_r; }
                const SliceRange* operator->() const { return m_r; }
                Iterator& operator++();

                Iterator operator++(int) = delete;

                friend bool operator==(const Iterator& lhs, const Iterator& rhs)
                {
                    return lhs.m_r == rhs.m_r;
                }
                friend bool operator!=(const Iterator& lhs, const Iterator& rhs)
                {
                    return !(lhs == rhs);
                }

            private:
                SliceRange* m_r;
            };

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
            const Strides m_strides;
            Coordinate m_coordinate;
        };

        inline SliceRange slice(const Shape& source_shape,
                                const Coordinate& source_start_corner,
                                const Coordinate& source_end_corner,
                                const Strides& strides)
        {
            return SliceRange{source_shape, source_start_corner, source_end_corner, strides};
        }

        class ReverseRange
        {
        public:
            ReverseRange(const Shape& source_shape, const AxisSet& reversed_axes)
                : m_source_shape{source_shape}
                , m_reversed_axes{reversed_axes}
                , m_coordinate(source_shape.size(), 0)
                , m_reversed_coordinate(source_shape.size(), 0)
            {
            }

            size_t index() const;

            class Iterator
            {
            public:
                using value_type = ReverseRange;
                using reference = ReverseRange&;
                using iterator_category = std::input_iterator_tag;
                using pointer = ReverseRange*;
                using difference_type = void;

                Iterator(ReverseRange* r)
                    : m_r{r}
                {
                }

                const ReverseRange& operator*() const { return *m_r; }
                const ReverseRange* operator->() const { return m_r; }
                Iterator& operator++();

                Iterator operator++(int) = delete;

                friend bool operator==(const Iterator& lhs, const Iterator& rhs)
                {
                    return lhs.m_r == rhs.m_r;
                }
                friend bool operator!=(const Iterator& lhs, const Iterator& rhs)
                {
                    return !(lhs == rhs);
                }

            private:
                ReverseRange* m_r;
            };

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
