// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
// This used to be lambda expressions but MSVC had difficulty compiling it. This way is more explicit.
template <bool D, typename T, typename U>
inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    if (std::get<0>(a) != std::get<0>(b)) {
        return D ? std::get<0>(a) > std::get<0>(b) : std::get<0>(a) < std::get<0>(b);
    } else {
        return std::get<1>(a) < std::get<1>(b);
    }
}

template <typename T, typename U>
inline bool compare_indices_ascending(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    return std::get<1>(a) < std::get<1>(b);
}

/**
 * @brief Reference implementation for TopK operator
 *
 * @param arg         Pointer to input data.
 * @param out_indices Pointer to output indicies.
 * @param out_values  Pointer to output values.
 * @param in_shape    Input data shape.
 * @param out_shape   Output data (values, indicies) shape.
 * @param axis        Axis for search of top K elements.
 * @param k           Number to find of top elements.
 * @param compute_max Select mode of find max or min.
 * @param sort        Sorting type.
 */
template <typename T,
          typename U,
          typename std::enable_if<std::is_same<typename std::decay<U>::type, int64_t>::value>::type* = nullptr>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const Shape& in_shape,
          const Shape& out_shape,
          const size_t axis,
          const size_t k,
          const bool compute_max,
          const op::TopKSortType sort = op::TopKSortType::NONE) {
    // Create temp vector for sorting.
    std::vector<std::tuple<T, U>> workspace(in_shape[axis]);
    const auto in_strides = row_major_strides(in_shape);
    const auto out_strides = row_major_strides(out_shape);
    const auto in_axis_stride = in_strides[axis];
    const auto out_axis_stride = out_strides[axis];

    const auto cmp_func = compute_max ? compare_max<true, T, U> : compare_max<false, T, U>;

    typename std::decay<decltype(cmp_func)>::type sort_func;
    switch (sort) {
    case op::TopKSortType::SORT_INDICES:
        sort_func = compare_indices_ascending<T, U>;
        break;
    case op::TopKSortType::SORT_VALUES:
        sort_func = cmp_func;
        break;
    default:
        sort_func = nullptr;
        break;
    }

    // Iterate over elements with 0 index at "axis" dimension
    auto traverse_shape = in_shape;
    traverse_shape[axis] = 1;
    CoordinateTransformBasic traverse_transform(traverse_shape);
    for (const auto& coord : traverse_transform) {
        auto arg_index = coordinate_index(coord, in_shape);
        auto out_index = coordinate_index(coord, out_shape);
        // Fill the temp vector
        U i = 0;
        for (auto& entry : workspace) {
            std::get<0>(entry) = arg[arg_index];
            std::get<1>(entry) = i;
            arg_index += in_axis_stride;
            ++i;
        }

        std::nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), cmp_func);
        if (sort_func) {
            std::sort(workspace.begin(), workspace.begin() + k, sort_func);
        }

        for (size_t j = 0; j < k; ++j) {
            const auto& entry = workspace[j];
            out_values[out_index] = std::get<0>(entry);
            out_indices[out_index] = std::get<1>(entry);
            out_index += out_axis_stride;
        }
    }
}

/**
 * @brief Reference implementation for TopK operator
 *
 * @param arg         Pointer to input data.
 * @param out_indices Pointer to output indicies.
 * @param out_values  Pointer to output values.
 * @param in_shape    Input data shape.
 * @param out_shape   Output data (values, indicies) shape.
 * @param axis        Axis for search of top K elements.
 * @param k           Number to find of top elements.
 * @param compute_max Select mode of find max or min.
 * @param sort        Sorting type.
 */
template <typename T,
          typename U,
          typename std::enable_if<!std::is_same<typename std::decay<U>::type, int64_t>::value>::type* = nullptr>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const Shape& in_shape,
          const Shape& out_shape,
          const size_t axis,
          const size_t k,
          const bool compute_max,
          const op::TopKSortType sort = op::TopKSortType::NONE) {
    const auto out_count = shape_size(out_shape);
    auto temp_out_indices = std::vector<int64_t>(out_count);

    topk(arg, temp_out_indices.data(), out_values, in_shape, out_shape, axis, k, compute_max, sort);

    for (auto i = out_count; i-- > 0;)
        out_indices[i] = static_cast<U>(temp_out_indices[i]);
}
}  // namespace reference
}  // namespace ov
