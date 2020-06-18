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
    /// \brief A useful class that allows to iterate over the tensor coordinates.
    ///        For example, for tensor with dimensions {2, 3} this iterator
    ///        produces the following coordinates:
    ///             {0,0}, {0,1}, {0,2},
    ///             {1,0}, {1,1}, {2,2}
    class NGRAPH_API CoordinateIterator
    {
    public:
        /// \brief Coordinates iterator constructor
        /// \param target_shape The target shape for coordinates iteration
        /// \param is_end The flag indicates that the coordinate iterator is the last.
        CoordinateIterator(const Shape& target_shape, bool is_end = false);

        /// \brief The postfix operation increment the iterator by one.
        void operator++();

        /// \brief The prefix operation increment the iterator by one.
        CoordinateIterator operator++(int);

        /// \brief Increments iterator n times.
        /// \param n number of elements it should be advanced
        void operator+=(size_t n);

        /// \brief Iterator dereferencing operator returns reference to current pointed coordinate.
        const Coordinate& operator*() const noexcept;

        /// \brief Checks for iterator inequality.
        /// \param it second iterator to compare
        bool operator!=(const CoordinateIterator& it) const noexcept;

        /// \brief Checks for iterator equality.
        /// \param it second iterator to compare
        bool operator==(const CoordinateIterator& it) const noexcept;

        /// \brief Increments iterator using specified axis of the shape n times.
        /// \param axis index used for iteration
        void advance(size_t axis) noexcept;

        /// \brief Useful function to build the last iterator.
        ///        Returns a singleton that points to the last iterator.
        static const CoordinateIterator& end();

    private:
        Shape m_target_shape;
        Coordinate m_coordinate;
        bool m_oob;
        bool m_empty;
    };

    /// \brief Class which allows to calculate item index with given coordinates in tensor
    ///        and helps to iterate over all coordinates.
    ///        Tensor items should be placed in memory in row-major order.
    class NGRAPH_API CoordinateTransformBasic
    {
    public:
        using Iterator = CoordinateIterator;

        CoordinateTransformBasic(const Shape& source_shape);

        /// \brief The tensor element index calculation by given coordinate.
        /// \param c tensor element coordinate
        size_t index(const Coordinate& c) const noexcept;

        /// \brief Returns an iterator to the first coordinate of the tensor.
        CoordinateIterator begin() const noexcept;

        /// \brief Returns an iterator to the coordinate following the last element of the tensor.
        const CoordinateIterator& end() const noexcept;

    protected:
        Shape m_source_shape;
    };

    /// \brief Class which allows to calculate item index with given coordinates in tensor
    ///        and helps to iterate over the subset of coordinates.
    ///        Tensor items should be placed in memory in row-major order.
    class NGRAPH_API CoordinateTransform : protected CoordinateTransformBasic
    {
    public:
        using Iterator = CoordinateIterator;

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

        /// \brief The tensor element index calculation by given coordinate.
        /// \param c tensor element coordinate
        size_t index(const Coordinate& c) const;

        /// \brief Checks that coordinate belongs to given coordinates subset.
        /// \param c tensor element coordinate
        bool has_source_coordinate(const Coordinate& c) const;

        /// \brief Convert a target-space coordinate to a source-space coordinate.
        /// \param c tensor element coordinate
        Coordinate to_source_coordinate(const Coordinate& c) const;

        const Shape& get_source_shape() const noexcept;
        const Shape& get_target_shape() const noexcept;
        const Coordinate& get_source_start_corner() const noexcept;
        const Coordinate& get_source_end_corner() const noexcept;
        const Strides& get_source_strides() const noexcept;
        const AxisVector& get_source_axis_order() const noexcept;
        const Strides& get_target_dilation_strides() const noexcept;

        /// \brief Returns an iterator to the first coordinate of the tensor.
        CoordinateIterator begin() const noexcept;

        /// \brief Returns an iterator to the coordinate following the last element of the tensor.
        const CoordinateIterator& end() const noexcept;

    private:
        Coordinate m_source_start_corner;
        Coordinate m_source_end_corner;
        Strides m_source_strides;
        AxisVector m_source_axis_order;
        CoordinateDiff m_target_padding_below;
        CoordinateDiff m_target_padding_above;
        Strides m_target_dilation_strides;

        Shape m_target_shape;
        size_t m_n_axes;
    };
}
