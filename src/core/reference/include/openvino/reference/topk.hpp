// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "openvino/op/topk.hpp"
#include "openvino/reference/utils/coordinate_index.hpp"
#include "openvino/reference/utils/coordinate_transform.hpp"

namespace ov {
namespace reference {
// This used to be lambda expressions but MSVC had difficulty compiling it. This way is more explicit.
template <bool D, typename T, typename U>
inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
// this is intentional to be able to compare floats directly
// without using relative or absolute tolerance
#if defined(__GNUC__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
    if (std::get<0>(a) == std::get<0>(b)) {
        return std::get<1>(a) < std::get<1>(b);
    }
#if defined(__GNUC__)
#    pragma GCC diagnostic pop
#endif

    if (D)
        return std::get<0>(a) > std::get<0>(b);
    else
        return std::get<0>(a) < std::get<0>(b);
}

template <typename T, typename U>
inline bool compare_indices_ascending(const std::tuple<T, U>& a, const std::tuple<T, U>& b) {
    return std::get<1>(a) < std::get<1>(b);
}

// TopK reference implementation provides stable indices output
template <typename T, typename U>
void topk(const T* arg,
          U* out_indices,
          T* out_values,
          const Shape& in_shape,
          const Shape& out_shape,
          size_t axis,
          size_t k,
          bool compute_max,
          op::TopKSortType sort = op::TopKSortType::NONE) {
    using namespace std;
    // Create temp vector for sorting.
    vector<tuple<T, U>> workspace(in_shape[axis]);
    vector<size_t> in_strides = row_major_strides(in_shape);
    vector<size_t> out_strides = row_major_strides(out_shape);
    auto in_axis_stride = in_strides[axis];
    auto out_axis_stride = out_strides[axis];

    // Iterate over elements with 0 index at "axis" dimension
    auto traverse_shape = in_shape;
    traverse_shape[axis] = 1;
    CoordinateTransformBasic traverse_transform(traverse_shape);
    for (const Coordinate& coord : traverse_transform) {
        auto arg_index = coordinate_index(coord, in_shape);
        auto out_index = coordinate_index(coord, out_shape);
        // Fill the temp vector
        U i = 0;
        for (tuple<T, U>& entry : workspace) {
            get<0>(entry) = arg[arg_index];
            get<1>(entry) = i;
            arg_index += in_axis_stride;
            i++;
        }
        // Sort the temp vector
        if (compute_max) {
            nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), compare_max<true, T, U>);
        } else {
            nth_element(workspace.begin(), workspace.begin() + k, workspace.end(), compare_max<false, T, U>);
        }
        // Write temp vector to output
        switch (sort) {
        case op::TopKSortType::NONE:
            break;
        case op::TopKSortType::SORT_INDICES:
            std::sort(workspace.begin(), workspace.begin() + k, compare_indices_ascending<T, U>);
            break;
        case op::TopKSortType::SORT_VALUES:
            if (compute_max)
                std::sort(workspace.begin(), workspace.begin() + k, compare_max<true, T, U>);
            else
                std::sort(workspace.begin(), workspace.begin() + k, compare_max<false, T, U>);
        }
        for (size_t j = 0; j < k; j++) {
            const auto& entry = workspace[j];
            out_values[out_index] = get<0>(entry);
            out_indices[out_index] = get<1>(entry);
            out_index += out_axis_stride;
        }
    }
}
}  // namespace reference
}  // namespace ov
