// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/op/topk.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // Had to split out these two functions. They used to be lambda expressions but
            // MSVC had difficulty compiling. This way is more explicit.
            template <typename T, typename U>
            inline bool compare_max(const std::tuple<T, U>& a, const std::tuple<T, U>& b)
            {
// this is intentional to be able to compare floats directly
// without using relative or absolute tolerance
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif
                if (std::get<0>(a) == std::get<0>(b))
                {
                    return std::get<1>(a) < std::get<1>(b);
                }
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
                return a > b;
            }

            template <typename T, typename U>
            inline bool compare_min(const std::tuple<T, U>& a, const std::tuple<T, U>& b)
            {
                return a < b;
            }

            template <typename T, typename U>
            inline bool sort_indices_ascending(const std::tuple<T, U>& a, const std::tuple<T, U>& b)
            {
                return std::get<1>(a) < std::get<1>(b);
            }

            template <typename T, typename U>
            void topk(const T* arg,
                      U* out_indices,
                      T* out_values,
                      const Shape& in_shape,
                      const Shape& out_shape,
                      size_t axis,
                      size_t k,
                      bool compute_max,
                      op::v1::TopK::SortType sort = op::v1::TopK::SortType::NONE)
            {
                using namespace std;
                // reorder source axis visit order and make "axis" inner most
                size_t ndim = static_cast<size_t>(in_shape.size());
                Coordinate start_corner(ndim, 0);
                Coordinate end_corner(in_shape);
                end_corner[axis] = 1;
                Strides strides(ndim, 1);
                AxisVector axis_order(ndim);
                iota(axis_order.begin(), axis_order.end(), 0);
                axis_order.erase(axis_order.begin() + axis);
                axis_order.push_back(axis);
                // Create CoordinateTransforms that visits only the first element along "axis"
                CoordinateTransform input_transform(
                    in_shape, start_corner, end_corner, strides, axis_order);
                CoordinateTransform output_transform(
                    out_shape, start_corner, end_corner, strides, axis_order);
                // Create temp vector for sorting.
                vector<tuple<T, U>> workspace(in_shape[axis]);
                vector<size_t> in_strides = ngraph::row_major_strides(in_shape);
                vector<size_t> out_strides = ngraph::row_major_strides(out_shape);
                auto in_axis_stride = in_strides[axis];
                auto out_axis_stride = out_strides[axis];
                for (const Coordinate& coord : input_transform)
                {
                    auto arg_index = input_transform.index(coord);
                    auto out_index = output_transform.index(coord);
                    // Fill the temp vector
                    U i = 0;
                    for (tuple<T, U>& entry : workspace)
                    {
                        get<0>(entry) = arg[arg_index];
                        get<1>(entry) = i;
                        arg_index += in_axis_stride;
                        i++;
                    }
                    // Sort the temp vector
                    if (compute_max)
                    {
                        nth_element(workspace.begin(),
                                    workspace.begin() + k,
                                    workspace.end(),
                                    compare_max<T, U>);
                    }
                    else
                    {
                        nth_element(workspace.begin(),
                                    workspace.begin() + k,
                                    workspace.end(),
                                    compare_min<T, U>);
                    }
                    // Write temp vector to output
                    switch (sort)
                    {
                    case op::v1::TopK::SortType::NONE: break;
                    case op::v1::TopK::SortType::SORT_INDICES:
                        std::sort(
                            workspace.begin(), workspace.begin() + k, sort_indices_ascending<T, U>);
                        break;
                    case op::v1::TopK::SortType::SORT_VALUES:
                        if (compute_max)
                            std::sort(workspace.begin(), workspace.begin() + k, compare_max<T, U>);
                        else
                            std::sort(workspace.begin(), workspace.begin() + k, compare_min<T, U>);
                    }
                    for (size_t j = 0; j < k; j++)
                    {
                        tuple<T, U> entry = workspace[j];
                        out_values[out_index] = get<0>(entry);
                        out_indices[out_index] = get<1>(entry);
                        out_index += out_axis_stride;
                    }
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
