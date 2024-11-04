// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/parallel.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
template <typename T, typename TOut = int64_t>
void search_sorted(const T* sorted,
                   const T* values,
                   TOut* out,
                   const Shape& sorted_shape,
                   const Shape& values_shape,
                   bool right_mode) {
    const CoordinateTransformBasic values_transform{values_shape};

    std::function<const T*(const T*, const T*, T)> compare_func = nullptr;
    if (right_mode) {
        compare_func = [](const T* begin, const T* end, T value) {
            return std::lower_bound(begin, end, value, std::less_equal<T>());
        };
    } else {
        compare_func = [](const T* begin, const T* end, T value) {
            return std::lower_bound(begin, end, value, std::less<T>());
        };
    }

    const size_t size = shape_size(values_shape);

    auto func = [&](size_t i) {
        auto it = values_transform.begin();
        it += i;
        const Coordinate& values_coord = *it;

        const auto values_index = coordinate_index(values_coord, values_shape);
        const T value = values[values_index];

        Coordinate sorted_coord_begin = values_coord;
        sorted_coord_begin.back() = 0;

        Coordinate sorted_coord_last = values_coord;
        sorted_coord_last.back() = sorted_shape.back();

        const auto sorted_index_begin = coordinate_index(sorted_coord_begin, sorted_shape);
        const auto sorted_index_last = coordinate_index(sorted_coord_last, sorted_shape);

        const T* idx_ptr = compare_func(sorted + sorted_index_begin, sorted + sorted_index_last, value);

        const ptrdiff_t sorted_index = (idx_ptr - sorted) - sorted_index_begin;

        out[values_index] = static_cast<TOut>(sorted_index);
    };

    ov::parallel_for(size, func);
}

}  // namespace reference
}  // namespace ov