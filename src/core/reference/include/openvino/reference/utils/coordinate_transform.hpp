// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

namespace ov {
/// \brief A useful class that allows to iterate over the tensor coordinates.
///        For example, for tensor with dimensions {2, 3} this iterator
///        produces the following coordinates:
///             {0,0}, {0,1}, {0,2},
///             {1,0}, {1,1}, {2,2}
class CoordinateIterator {
    /// \brief Coordinates iterator constructor
    /// \param target_shape The target shape for coordinates iteration
    /// \param is_end The flag indicates that the coordinate iterator is the last.
    CoordinateIterator(const Shape& target_shape, bool is_end);

public:
    /// \brief Coordinates iterator constructor
    /// \param target_shape The target shape for coordinates iteration
    CoordinateIterator(const Shape& target_shape) : CoordinateIterator(target_shape, false) {}

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
    size_t advance(size_t axis) noexcept;

    /// \brief Useful function to build the last iterator.
    ///        Returns a singleton that points to the last iterator.
    static const CoordinateIterator& end();

private:
    const Shape& m_target_shape;
    Coordinate m_coordinate;
    bool m_oob;
};

/// \brief Class which allows to calculate item index with given coordinates in tensor
///        and helps to iterate over all coordinates.
///        Tensor items should be placed in memory in row-major order.
class CoordinateTransformBasic {
public:
    using Iterator = CoordinateIterator;

    CoordinateTransformBasic(const Shape& source_shape);

    /// \brief Returns an iterator to the first coordinate of the tensor.
    CoordinateIterator begin() const noexcept;

    /// \brief Returns an iterator to the coordinate following the last element of the tensor.
    const CoordinateIterator& end() const noexcept;

protected:
    Shape m_source_shape;
};
}  // namespace ov
