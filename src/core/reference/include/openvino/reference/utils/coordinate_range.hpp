// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <iterator>

#include "openvino/core/coordinate.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/strides.hpp"

namespace ov {
namespace coordinates {
namespace impl {
namespace {
template <typename C>
bool has_zeros(const C& c) {
    const auto is_zero = [](size_t x) {
        return x == 0;
    };
    return std::any_of(c.begin(), c.end(), is_zero);
}

}  // namespace

/// \brief Class which allow to iterate over a different ranges part by part
///
template <typename Range>
class RangeIterator {
public:
    using value_type = typename Range::value_type;
    using reference = typename Range::value_type;
    using iterator_category = std::input_iterator_tag;
    using difference_type = void;

    RangeIterator(Range* r) : m_r{r} {
        if (m_r && !m_r->is_valid()) {
            m_r = nullptr;
        }
    }

    value_type operator*() const {
        return m_r->get_value();
    }
    RangeIterator& operator++() {
        if (m_r && !m_r->increment()) {
            m_r = nullptr;
        }
        return *this;
    }

    RangeIterator operator++(int) = delete;

    friend bool operator==(const RangeIterator& lhs, const RangeIterator& rhs) {
        return lhs.m_r == rhs.m_r;
    }
    friend bool operator!=(const RangeIterator& lhs, const RangeIterator& rhs) {
        return !(lhs == rhs);
    }

private:
    Range* m_r;
};

/// \brief Describe slice range
///
struct CoordinateBounds {
    CoordinateBounds(const Coordinate& lower, const Coordinate& upper) : m_lower{lower}, m_upper{upper} {
        if (m_lower.size() != m_upper.size()) {
            throw std::domain_error{"different Coordinates bonds sizes"};
        }
    }
    Coordinate m_lower;
    Coordinate m_upper;

    size_t last_dim_size() const noexcept {
        return m_upper.back() - m_lower.back();
    }
};

/// \brief helper for iterator creation which allow to stay DRY
///
template <typename Range>
struct RangeBase {
    using Iterator = RangeIterator<Range>;

    Iterator begin() {
        return Iterator(static_cast<Range*>(this));
    }
    Iterator end() {
        return Iterator(nullptr);
    }
    friend Iterator begin(Range& r) {
        return r.begin();
    }
    friend Iterator end(Range& r) {
        return r.end();
    }
};

/// \brief Information how index in _Range_ should be change
///
enum class Direction {
    forward,
    reverse,
};

/// \brief Range contain information which part of memory should be copied
///
struct Range {
    const size_t begin_index;
    const size_t element_number;
    const size_t step;
    const Direction direction;

    static constexpr Range make_empyt() {
        return Range{0, 0, 1, Direction::forward};
    }
};

/// \brief Class allows to iterate over sliced Tensor part by part.
///
/// To create SliceRange use _slice_ function.
class SliceRange : public RangeBase<SliceRange> {
public:
    using value_type = Range;
    SliceRange(const Shape& source_shape,
               const Coordinate& source_start_corner,
               const Coordinate& source_end_corner,
               const Strides& strides);

    value_type get_value() const;

    bool increment();

    bool is_valid() const noexcept {
        return !has_zeros(m_source_shape);
    }

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
                        const Strides& strides) {
    return SliceRange{source_shape, source_start_corner, source_end_corner, strides};
}

/// \brief Create SliceRange which might be used in range-base for loop
///
inline SliceRange slice(const Shape& source_shape,
                        const Coordinate& source_start_corner,
                        const Coordinate& source_end_corner) {
    return slice(source_shape, source_start_corner, source_end_corner, Strides(source_shape.size(), 1));
}

/// \brief Class allows to iterate over Tensor with reverted axes part by part.
///
/// To create ReverseRange use _reverse_ function.
///
class ReverseRange : public RangeBase<ReverseRange> {
public:
    using value_type = Range;
    ReverseRange(const Shape& source_shape, const AxisSet& reversed_axis);

    value_type get_value() const;

    bool increment();

    bool is_valid() const noexcept {
        return !has_zeros(m_source_shape);
    }

private:
    const Shape m_source_shape;
    const std::vector<size_t> m_memory_strides;
    const std::vector<Direction> m_axis_directions;
    Coordinate m_coordinate;
    size_t m_index{0};
};

inline ReverseRange reverse(const Shape& source_shape, const AxisSet& reversed_axis) {
    return ReverseRange(source_shape, reversed_axis);
}

inline ReverseRange index(const Shape& source_shape) {
    return reverse(source_shape, {});
}

}  // namespace impl
using impl::Direction;
using impl::index;
using impl::reverse;
using impl::slice;
}  // namespace coordinates
}  // namespace ov
