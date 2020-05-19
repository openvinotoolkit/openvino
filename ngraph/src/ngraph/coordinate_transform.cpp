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

#include <cstdio>
#include <iostream>
#include <sstream>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_diff.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/except.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/strides.hpp"
#include "ngraph/util.hpp"

using namespace ngraph;

CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides,
                                         const AxisVector& source_axis_order,
                                         const CoordinateDiff& target_padding_below,
                                         const CoordinateDiff& target_padding_above,
                                         const Strides& target_dilation_strides)
    : m_source_shape(source_shape)
    , m_source_start_corner(source_start_corner)
    , m_source_end_corner(source_end_corner)
    , m_source_strides(source_strides)
    , m_source_axis_order(source_axis_order)
    , m_target_padding_below(target_padding_below)
    , m_target_padding_above(target_padding_above)
    , m_target_dilation_strides(target_dilation_strides)
    , m_end_iterator(Shape(), true)
{
    m_n_axes = source_shape.size();

    if (m_n_axes != source_start_corner.size())
    {
        throw std::domain_error(
            "Source start corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_end_corner.size())
    {
        throw std::domain_error(
            "Source end corner does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_strides.size())
    {
        throw std::domain_error(
            "Source strides do not have the same number of axes as the source space shape");
    }
    if (m_n_axes != source_axis_order.size())
    {
        // Note: this check is NOT redundant with the is_permutation check below, though you might
        // think it is. If the lengths don't match then is_permutation won't catch that; it'll
        // either stop short or walk off the end of source_axis_order.
        throw std::domain_error(
            "Source axis order does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_padding_below.size())
    {
        throw std::domain_error(
            "Padding-below shape does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_padding_above.size())
    {
        throw std::domain_error(
            "Padding-above shape does not have the same number of axes as the source space shape");
    }
    if (m_n_axes != target_dilation_strides.size())
    {
        throw std::domain_error(
            "Target dilation strides do not have the same number of axes as the source shape");
    }

    AxisVector all_axes(m_n_axes);
    for (size_t i = 0; i < all_axes.size(); i++)
    {
        all_axes[i] = i;
    }

    if (!std::is_permutation(all_axes.begin(), all_axes.end(), source_axis_order.begin()))
    {
        throw std::domain_error(
            "Source axis order is not a permutation of {0,...,n-1} where n is the number of axes "
            "in the source space shape");
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (target_dilation_strides[i] == 0)
        {
            std::stringstream ss;

            ss << "The target dilation stride is 0 at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    std::vector<std::ptrdiff_t> padded_upper_bounds;

    for (size_t i = 0; i < m_n_axes; i++)
    {
        std::ptrdiff_t padded_upper_bound =
            subtract_or_zero(source_shape[i], size_t(1)) * target_dilation_strides[i] + 1 +
            target_padding_below[i] + target_padding_above[i];

        if (padded_upper_bound < 0)
        {
            std::stringstream ss;

            ss << "The end corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }

        padded_upper_bounds.push_back(padded_upper_bound);
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (static_cast<int64_t>(source_start_corner[i]) >= padded_upper_bounds[i] &&
            source_start_corner[i] != source_shape[i])
        {
            std::stringstream ss;

            ss << "The start corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }

        if (static_cast<int64_t>(source_end_corner[i]) > padded_upper_bounds[i])
        {
            std::stringstream ss;

            ss << "The end corner is out of bounds at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t i = 0; i < m_n_axes; i++)
    {
        if (source_strides[i] == 0)
        {
            std::stringstream ss;

            ss << "The source stride is 0 at axis " << i;
            throw std::domain_error(ss.str());
        }
    }

    for (size_t axis = 0; axis < m_n_axes; axis++)
    {
        m_target_shape.push_back(ceil_div(source_end_corner[source_axis_order[axis]] -
                                              source_start_corner[source_axis_order[axis]],
                                          source_strides[source_axis_order[axis]]));
    }
}

Strides CoordinateTransform::default_strides(size_t n_axes)
{
    return Strides(n_axes, 1);
}

CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides,
                                         const AxisVector& source_axis_order,
                                         const CoordinateDiff& target_padding_below,
                                         const CoordinateDiff& target_padding_above)
    : CoordinateTransform(source_shape,
                          source_start_corner,
                          source_end_corner,
                          source_strides,
                          source_axis_order,
                          target_padding_below,
                          target_padding_above,
                          default_strides(source_shape.size()))
{
}

CoordinateDiff CoordinateTransform::default_padding(size_t n_axes)
{
    return CoordinateDiff(n_axes, 0);
}

CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides,
                                         const AxisVector& source_axis_order)
    : CoordinateTransform(source_shape,
                          source_start_corner,
                          source_end_corner,
                          source_strides,
                          source_axis_order,
                          default_padding(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_strides(source_shape.size()))
{
}

AxisVector CoordinateTransform::default_axis_order(size_t n_axes)
{
    AxisVector result(n_axes);
    size_t n = 0;
    std::generate(result.begin(), result.end(), [&n]() -> size_t { return n++; });

    return result;
}

CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner,
                                         const Strides& source_strides)
    : CoordinateTransform(source_shape,
                          source_start_corner,
                          source_end_corner,
                          source_strides,
                          default_axis_order(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_strides(source_shape.size()))
{
}

CoordinateTransform::CoordinateTransform(const Shape& source_shape,
                                         const Coordinate& source_start_corner,
                                         const Coordinate& source_end_corner)
    : CoordinateTransform(source_shape,
                          source_start_corner,
                          source_end_corner,
                          default_strides(source_shape.size()),
                          default_axis_order(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_strides(source_shape.size()))
{
}

Coordinate CoordinateTransform::default_source_start_corner(size_t n_axes)
{
    return Coordinate(n_axes, 0);
}

Coordinate CoordinateTransform::default_source_end_corner(const Shape& source_shape)
{
    return source_shape;
}

CoordinateTransform::CoordinateTransform(const Shape& source_shape)
    : CoordinateTransform(source_shape,
                          default_source_start_corner(source_shape.size()),
                          default_source_end_corner(source_shape),
                          default_strides(source_shape.size()),
                          default_axis_order(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_padding(source_shape.size()),
                          default_strides(source_shape.size()))
{
}

// Compute the index of a source-space coordinate in the buffer.
size_t CoordinateTransform::index_source(const Coordinate& c) const
{
    size_t index = 0;
    size_t stride = 1;

    for (size_t axis = m_n_axes; axis-- > 0;)
    {
        index += c[axis] * stride;
        stride *= m_source_shape[axis];
    }

    return index;
}

// Compute the index of a target-space coordinate in thebuffer.
size_t CoordinateTransform::index(const Coordinate& c) const
{
    return index_source(to_source_coordinate(c));
}

// Convert a target-space coordinate to a source-space coordinate.
Coordinate CoordinateTransform::to_source_coordinate(const Coordinate& c_target) const
{
    if (c_target.size() != m_n_axes)
    {
        throw std::domain_error(
            "Target coordinate rank does not match the coordinate transform rank");
    }

    Coordinate c_source(c_target.size());

    for (size_t target_axis = 0; target_axis < m_n_axes; target_axis++)
    {
        size_t source_axis = m_source_axis_order[target_axis];

        size_t target_pos = c_target[target_axis];
        size_t pos_destrided = target_pos * m_source_strides[source_axis];
        size_t pos_deshifted = pos_destrided + m_source_start_corner[source_axis];
        size_t pos_depadded = pos_deshifted - m_target_padding_below[target_axis];
        size_t pos_dedilated = pos_depadded / m_target_dilation_strides[target_axis];
        c_source[source_axis] = pos_dedilated;
    }

    return c_source;
}

// A point in the target space is considered not to have a source coordinate if it was inserted due
// to padding or dilation, or if it is out of the bounds of the target space.
bool CoordinateTransform::has_source_coordinate(const Coordinate& c_target) const
{
    if (c_target.size() != m_n_axes)
    {
        throw std::domain_error(
            "Target coordinate rank does not match the coordinate transform rank");
    }

    for (size_t target_axis = 0; target_axis < m_n_axes; target_axis++)
    {
        // Is this coordinate out of bounds of the target space?
        if (c_target[target_axis] >= m_target_shape[target_axis])
        {
            return false;
        }

        // The rest of this is a replay of the corresponding logic in `to_source_coordinate`, with
        // bounds and divisibility checking.
        std::ptrdiff_t source_axis = m_source_axis_order[target_axis];

        std::ptrdiff_t target_pos = c_target[target_axis];
        std::ptrdiff_t pos_destrided = target_pos * m_source_strides[source_axis];
        std::ptrdiff_t pos_deshifted = pos_destrided + m_source_start_corner[source_axis];

        // If we are in the below-padding or the above-padding.
        if (pos_deshifted < m_target_padding_below[target_axis])
        {
            return false;
        }
        std::ptrdiff_t pos_depadded = pos_deshifted - m_target_padding_below[target_axis];

        // If we are in the above-padding, we have no source coordinate.
        if (m_source_shape[source_axis] == 0 ||
            (pos_depadded >= ((static_cast<int64_t>(m_source_shape[source_axis]) - 1) *
                              static_cast<int64_t>(m_target_dilation_strides[target_axis])) +
                                 1))
        {
            return false;
        }

        // If we are in a dilation gap, we have no source coordinate.
        if (pos_depadded % m_target_dilation_strides[target_axis] != 0)
        {
            return false;
        }
    }

    return true;
}

const Shape& CoordinateTransform::get_target_shape() const
{
    return m_target_shape;
}

// The "is_end" parameter is true if we want the "end()" iterator.
CoordinateTransform::Iterator::Iterator(const Shape& target_shape, bool is_end)
    : m_target_shape(target_shape)
{
    // Initial coordinate is (0,...,0) in the target space.
    m_coordinate = Coordinate(target_shape.size(), 0);

    // The case where we have a zero-length axis is a bit special, in that
    // the iterator always starts out of bounds.
    m_empty = false;

    for (auto s : target_shape)
    {
        if (s == 0)
        {
            m_empty = true;
            break;
        }
    }

    m_oob = is_end || m_empty;
}

void CoordinateTransform::Iterator::operator++()
{
    // If we are out of bounds, start over at (0,...0). (TODO: not sure if that's what we want. It
    // might be best to stay put?)
    if (m_oob)
    {
        std::fill(m_coordinate.begin(), m_coordinate.end(), 0);
        m_oob = m_empty;
        return;
    }

    // Increment the target coordinate.
    for (size_t axis = m_target_shape.size(); axis-- > 0;)
    {
        m_coordinate[axis]++;

        if (m_coordinate[axis] < m_target_shape[axis])
        {
            // No carry-out, so we are done.
            return;
        }
        else
        {
            m_coordinate[axis] = 0;
        }
    }

    // If we are still here there was carry-out from the most significant axis. We are now out of
    // bounds.
    m_oob = true;
}

CoordinateTransform::Iterator CoordinateTransform::Iterator::operator++(int)
{
    CoordinateTransform::Iterator temp = *this;
    ++(*this);
    return temp;
}

void CoordinateTransform::Iterator::operator+=(size_t n)
{
    for (size_t i = 0; i < n; i++)
    {
        ++(*this);
    }
}

const Coordinate& CoordinateTransform::Iterator::operator*() const
{
    return m_coordinate;
}

bool CoordinateTransform::Iterator::operator!=(const Iterator& it)
{
    return !(*this == it);
}

bool CoordinateTransform::Iterator::operator==(const Iterator& it)
{
    if (it.m_oob)
    {
        // Out-of-bounds iterators are always equal; in other words, an iterator is always equal to
        // end() even if the internally stored coordinates are different.

        // If one iterator is out of bounds and the other is not, they are unequal even if their
        // target coordinates happen to match.
        return m_oob;
    }
    else if (m_oob)
    {
        return false;
    }

    if (m_target_shape != it.m_target_shape)
    {
        return false;
    }

    // Check axis-wise if the iterators are on the same target coordinate.
    for (size_t axis = 0; axis < m_target_shape.size(); axis++)
    {
        if (m_coordinate[axis] != it.m_coordinate[axis])
        {
            return false;
        }
    }

    return true;
}
