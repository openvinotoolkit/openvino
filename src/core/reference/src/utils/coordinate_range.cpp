// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/utils/coordinate_range.hpp"

#include <cassert>
#include <numeric>
#include <stdexcept>

#include "openvino/reference/utils/coordinate_index.hpp"

namespace ov {
namespace coordinates {
namespace impl {
namespace {
std::vector<size_t> memory_strides(const Shape& shape) {
    std::vector<size_t> mem_strides(shape.size(), 1);

    if (shape.size() > 1) {
        for (auto i = shape.size() - 1; i-- > 0;) {
            mem_strides[i] = mem_strides[i + 1] * shape[i + 1];
        }
    }

    return mem_strides;
}

}  // namespace

SliceRange::SliceRange(const Shape& source_shape,
                       const Coordinate& source_start_corner,
                       const Coordinate& source_end_corner,
                       const Strides& source_strides)
    : m_source_shape{source_shape},
      m_bounds{source_start_corner, source_end_corner},
      m_source_strides{source_strides},
      m_memory_strides(memory_strides(source_shape)),
      m_coordinate{source_start_corner},
      m_index(coordinate_index(source_start_corner, source_shape)) {
    const auto axis = m_source_shape.size();

    if (axis != m_bounds.m_lower.size()) {
        throw std::domain_error("Source start corner does not have the same number of axis as the "
                                "source "
                                "space "
                                "shape");
    }
    if (axis != m_bounds.m_upper.size()) {
        throw std::domain_error("Source end corner does not have the same number of axis as the source "
                                "space "
                                "shape");
    }
    if (axis != m_source_strides.size()) {
        throw std::domain_error("Source strides do not have the same number of axis as the source "
                                "space "
                                "shape");
    }
    if (axis != m_memory_strides.size()) {
        throw std::runtime_error("Something goes wrong");
    }
}

SliceRange::value_type SliceRange::get_value() const {
    if (m_source_shape.empty()) {
        return Range::make_empyt();
    }
    const size_t element_no = (m_bounds.last_dim_size() + m_source_strides.back() - 1) / m_source_strides.back();

    return Range{m_index, element_no, m_source_strides.back(), Direction::forward};
}

bool SliceRange::increment() {
    // during increment rage omit last dim so at least two dims are required to proceed
    if (m_coordinate.size() < 2) {
        return false;
    }
    // omit last dim - it will be return in slice_range
    for (auto axis = m_coordinate.size() - 1; axis-- > 0;) {
        const auto index_step = m_source_strides[axis] * m_memory_strides[axis];
        m_coordinate[axis] += m_source_strides[axis];
        m_index += index_step;
        if (m_coordinate[axis] < m_bounds.m_upper[axis]) {
            assert(m_index < shape_size(m_source_shape));
            return true;
        }
        const auto difference = m_coordinate[axis] - m_bounds.m_lower[axis];
        m_coordinate[axis] = m_bounds.m_lower[axis];

        // back on beginning of axis memory
        m_index -= difference * m_memory_strides[axis];
    }

    return false;
}

namespace {
std::vector<Direction> axis_direcions(size_t size, const AxisSet& reversed_axis) {
    const auto max_reversed_axis = [&] {
        return *std::max_element(reversed_axis.begin(), reversed_axis.end());
    };
    if (!reversed_axis.empty() && !(max_reversed_axis() < size)) {
        throw std::domain_error("Reversed axis have axes above the source space shape");
    }

    std::vector<Direction> directions(size, Direction::forward);
    for (auto i : reversed_axis) {
        directions[i] = Direction::reverse;
    }
    return directions;
}

Coordinate start_coordinate(const Shape& s, const std::vector<Direction>& direction) {
    Coordinate coordiante(s.size(), 0);
    for (size_t i = 0; i < s.size(); ++i) {
        if (direction[i] == Direction::reverse) {
            coordiante[i] = s[i] - 1;
        }
    }
    return coordiante;
}

}  // namespace

ReverseRange::ReverseRange(const Shape& source_shape, const AxisSet& reversed_axis)
    : m_source_shape{source_shape},
      m_memory_strides(memory_strides(source_shape)),
      m_axis_directions(axis_direcions(source_shape.size(), reversed_axis)),
      m_coordinate(source_shape.size(), 0),
      m_index(coordinate_index(start_coordinate(source_shape, m_axis_directions), source_shape)) {}

ReverseRange::value_type ReverseRange::get_value() const {
    if (m_source_shape.empty()) {
        return Range::make_empyt();
    }

    assert(m_memory_strides.back() == 1);
    return Range{m_index, m_source_shape.back(), m_memory_strides.back(), m_axis_directions.back()};
}

bool ReverseRange::increment() {
    // during increment rage omit last dim so at least two dims are required to proceed
    if (m_coordinate.size() < 2) {
        return false;
    }
    // omit last dim - it will be return in reverse_range
    for (auto axis = m_coordinate.size() - 1; axis-- > 0;) {
        const auto index_step = m_memory_strides[axis];
        ++m_coordinate[axis];
        if (m_axis_directions[axis] == Direction::forward) {
            m_index += index_step;
        } else {
            m_index -= index_step;
        }
        if (m_coordinate[axis] < m_source_shape[axis]) {
            assert(0 <= m_index && m_index < shape_size(m_source_shape));
            return true;
        }
        m_coordinate[axis] = 0;

        // back on beginning of axis memory
        if (m_axis_directions[axis] == Direction::forward) {
            m_index -= m_source_shape[axis] * m_memory_strides[axis];
        } else {
            m_index += m_source_shape[axis] * m_memory_strides[axis];
        }
    }
    return false;
}

}  // namespace impl

}  // namespace coordinates
}  // namespace ov
