// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/coordinate_transform.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"

using namespace ov;

CoordinateTransformBasic::CoordinateTransformBasic(const Shape& source_shape) : m_source_shape(source_shape) {}

CoordinateIterator CoordinateTransformBasic::begin() const noexcept {
    return CoordinateIterator(m_source_shape);
}

const CoordinateIterator& CoordinateTransformBasic::end() const noexcept {
    return CoordinateIterator::end();
}

// The "is_end" parameter is true if we want the "end()" iterator.
CoordinateIterator::CoordinateIterator(const Shape& target_shape, bool is_end)
    : m_target_shape(target_shape),
      m_coordinate(target_shape.size(), 0) {
    // The case where we have a zero-length axis is a bit special, in that
    // the iterator always starts out of bounds.
    bool const empty = std::find(target_shape.begin(), target_shape.end(), 0) != target_shape.end();

    m_oob = is_end || empty;
}

void CoordinateIterator::operator++() {
    advance(m_target_shape.size() - 1);
}

size_t CoordinateIterator::advance(size_t axis) noexcept {
    m_oob |= m_target_shape.empty();

    if (m_oob)
        return m_target_shape.size();

    // Increment the target coordinate.
    do {
        m_coordinate[axis]++;

        if (m_coordinate[axis] < m_target_shape[axis]) {
            // No carry-out, so we are done.
            return axis;
        } else {
            m_coordinate[axis] = 0;
        }
    } while (axis-- > 0);

    // If we are still here there was carry-out from the most significant axis. We are now out of
    // bounds.
    m_oob = true;

    return m_target_shape.size();
}

CoordinateIterator CoordinateIterator::operator++(int) {
    CoordinateIterator temp = *this;
    ++(*this);
    return temp;
}

void CoordinateIterator::operator+=(size_t n) {
    for (size_t i = 0; i < n; i++) {
        ++(*this);
    }
}

const Coordinate& CoordinateIterator::operator*() const noexcept {
    return m_coordinate;
}

bool CoordinateIterator::operator!=(const CoordinateIterator& it) const noexcept {
    return !(*this == it);
}

bool CoordinateIterator::operator==(const CoordinateIterator& it) const noexcept {
    if (it.m_oob) {
        // Out-of-bounds iterators are always equal; in other words, an iterator is always equal to
        // end() even if the internally stored coordinates are different.

        // If one iterator is out of bounds and the other is not, they are unequal even if their
        // target coordinates happen to match.
        return m_oob;
    } else if (m_oob) {
        return false;
    }

    if (m_target_shape != it.m_target_shape) {
        return false;
    }

    // Check axis-wise if the iterators are on the same target coordinate.
    for (size_t axis = 0; axis < m_target_shape.size(); axis++) {
        if (m_coordinate[axis] != it.m_coordinate[axis]) {
            return false;
        }
    }

    return true;
}

const CoordinateIterator& CoordinateIterator::end() {
    static const CoordinateIterator it(Shape(), true);
    return it;
}
