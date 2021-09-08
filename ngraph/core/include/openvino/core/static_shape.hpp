// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <numeric>
#include <vector>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/axis_set.hpp"
#include "openvino/core/core_visibility.hpp"
#include "openvino/core/strides.hpp"

namespace ov {
/// \brief StaticShape for a tensor.
class StaticShape : public std::vector<size_t> {
public:
    OPENVINO_API StaticShape();

    OPENVINO_API StaticShape(const std::initializer_list<size_t>& axis_lengths);

    OPENVINO_API StaticShape(const std::vector<size_t>& axis_lengths);

    OPENVINO_API StaticShape(const StaticShape& axis_lengths);

    OPENVINO_API explicit StaticShape(size_t n, size_t initial_value = 0);

    OPENVINO_API ~StaticShape();

    template <class InputIterator>
    StaticShape(InputIterator first, InputIterator last) : std::vector<size_t>(first, last) {}

    OPENVINO_API StaticShape& operator=(const StaticShape& v);
    OPENVINO_API StaticShape& operator=(StaticShape&& v) noexcept;
};

/// Number of elements in spanned by a shape
template <typename SHAPE_TYPE>
size_t shape_size(const SHAPE_TYPE& shape) {
    size_t size = 1;
    for (auto d : shape) {
        size *= d;
    }
    return size;
}

/// Number of elements in a subset of dimensions of a shape.
/// Returns a product of dimensions in a range [start_dim;end_dim)
template <typename ForwardIt>
size_t shape_size(ForwardIt start_dim, const ForwardIt end_dim) {
    static_assert(std::is_arithmetic<typename std::iterator_traits<ForwardIt>::value_type>::value,
                  "shape_size expects 2 forward iterators as inputs. value_type of those iterators has to be an "
                  "arithmetic type so that they can be used in multiplication operation.");

    return std::accumulate(start_dim,
                           end_dim,
                           typename std::iterator_traits<ForwardIt>::value_type{1},
                           std::multiplies<typename std::iterator_traits<ForwardIt>::value_type>());
}

/// Row-major strides for a shape
template <typename SHAPE_TYPE>
std::vector<size_t> row_major_strides(const SHAPE_TYPE& shape) {
    std::vector<size_t> strides(shape.size());
    size_t s = 1;
    auto st = strides.rbegin();
    for (auto d = shape.rbegin(); d != shape.rend() && st != strides.rend(); d++, st++) {
        *st = s;
        s *= *d;
    }
    return strides;
}

template <typename SHAPE_TYPE>
size_t row_major_stride(const SHAPE_TYPE& shape, size_t axis) {
    size_t s = 1;
    for (size_t i = shape.size(); i-- > axis + 1;) {
        s *= shape[i];
    }
    return s;
}

template <typename SHAPE_TYPE>
inline bool is_scalar(const SHAPE_TYPE& shape) {
    return 0 == shape.size();
}

template <typename SHAPE_TYPE>
inline bool is_vector(const SHAPE_TYPE& shape) {
    return 1 == shape.size();
}

OPENVINO_API
std::ostream& operator<<(std::ostream& s, const StaticShape& shape);

template <>
class OPENVINO_API AttributeAdapter<ov::StaticShape>
    : public IndirectVectorValueAccessor<ov::StaticShape, std::vector<int64_t>>

{
public:
    AttributeAdapter(ov::StaticShape& value)
        : IndirectVectorValueAccessor<ov::StaticShape, std::vector<int64_t>>(value) {}
    static constexpr DiscreteTypeInfo type_info{"AttributeAdapter<StaticShape>", 0};
    const DiscreteTypeInfo& get_type_info() const override {
        return type_info;
    }
};
}  // namespace ov
