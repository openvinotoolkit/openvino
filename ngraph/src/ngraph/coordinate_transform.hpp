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

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"

namespace ngraph
{
    class NGRAPH_API CoordinateTransform
    {
    public:
        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order,
                            const CoordinateDiff& target_padding_below,
                            const CoordinateDiff& target_padding_above,
                            const Strides& source_dilation_strides);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order,
                            const CoordinateDiff& target_padding_below,
                            const CoordinateDiff& target_padding_above);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides,
                            const AxisVector& source_axis_order);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner,
                            const Strides& source_strides);

        CoordinateTransform(const Shape& source_shape,
                            const Coordinate& source_start_corner,
                            const Coordinate& source_end_corner);

        CoordinateTransform(const Shape& source_shape);

        size_t index(const Coordinate& c) const;
        bool has_source_coordinate(const Coordinate& c) const;
        Coordinate to_source_coordinate(const Coordinate& c) const;
        const Shape& get_target_shape() const;

        const Shape& get_source_shape() const { return m_source_shape; }
        const Coordinate& get_source_start_corner() const { return m_source_start_corner; }
        const Coordinate& get_source_end_corner() const { return m_source_end_corner; }
        const Strides& get_source_strides() const { return m_source_strides; }
        const AxisVector& get_source_axis_order() const { return m_source_axis_order; }
        const Strides& get_target_dilation_strides() const { return m_target_dilation_strides; }
        class NGRAPH_API Iterator
        {
        public:
            Iterator(const Shape& target_shape, bool is_end = false);

            void operator++();
            Iterator operator++(int);
            void operator+=(size_t n);
            const Coordinate& operator*() const;
            bool operator!=(const Iterator& it);
            bool operator==(const Iterator& it);

        private:
            Shape m_target_shape;
            Shape m_axis_walk_order;
            Coordinate m_coordinate;
            bool m_oob;
            bool m_empty;
        };

        Iterator begin() noexcept { return Iterator(m_target_shape); }
        Iterator end() noexcept { return m_end_iterator; }
        size_t index_source(const Coordinate& c) const;
        static Strides default_strides(size_t n_axes);
        static CoordinateDiff default_padding(size_t n_axes);
        static AxisVector default_axis_order(size_t n_axes);
        static Coordinate default_source_start_corner(size_t n_axes);
        static Coordinate default_source_end_corner(const Shape& source_shape);

        Shape m_source_shape;
        Coordinate m_source_start_corner;
        Coordinate m_source_end_corner;
        Strides m_source_strides;
        AxisVector m_source_axis_order;
        CoordinateDiff m_target_padding_below;
        CoordinateDiff m_target_padding_above;
        Strides m_target_dilation_strides;

        Shape m_target_shape;
        size_t m_n_axes;
        Iterator m_end_iterator;
    };
}
