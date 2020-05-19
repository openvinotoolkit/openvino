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

#pragma once

#include <cstring>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void scatter_add(T* inputs,
                             U* indices,
                             T* updates,
                             T* out,
                             const Shape& inputs_shape,
                             const Shape& indices_shape,
                             const Shape& updates_shape,
                             const Shape& out_shape)
            {
                using namespace std;
                // Copy inputs to out
                memcpy(out, inputs, sizeof(T) * shape_size(inputs_shape));
                // Create a CoordinateTransform for "indices"
                size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                Coordinate indices_start_corner(indices_ndim, 0);
                Coordinate indices_end_corner(indices_shape);
                Strides indices_strides(indices_ndim, 1);
                AxisVector indices_axis_order(indices_ndim);
                iota(indices_axis_order.begin(), indices_axis_order.end(), 0);
                CoordinateTransform indices_transform(indices_shape,
                                                      indices_start_corner,
                                                      indices_end_corner,
                                                      indices_strides,
                                                      indices_axis_order);
                // Create an outer CoordinateTransform for "update"
                size_t updates_ndim = static_cast<size_t>(updates_shape.size());
                Coordinate updates_outer_start_corner(updates_ndim, 0);
                Coordinate updates_outer_end_corner(updates_shape);
                for (size_t i = indices_ndim; i < updates_ndim; i++)
                {
                    updates_outer_end_corner[i] = 1;
                }
                Strides updates_strides(updates_ndim, 1);
                AxisVector updates_axis_order(updates_ndim);
                iota(updates_axis_order.begin(), updates_axis_order.end(), 0);
                CoordinateTransform updates_outer_transform(updates_shape,
                                                            updates_outer_start_corner,
                                                            updates_outer_end_corner,
                                                            updates_strides,
                                                            updates_axis_order);
                // Common vars for out
                size_t out_ndim = static_cast<size_t>(out_shape.size());
                Strides out_strides(out_ndim, 1);
                AxisVector out_axis_order(out_ndim);
                iota(out_axis_order.begin(), out_axis_order.end(), 0);
                // Visit one updates silce and one out silce at a time.
                auto updates_outer_coord_iter = updates_outer_transform.begin();
                for (const Coordinate& indices_coord : indices_transform)
                {
                    auto indices_index = indices_transform.index(indices_coord);
                    U slice_index = indices[indices_index];
                    // Create CoordinateTransform for out slice
                    Coordinate out_start_corner(out_ndim, 0);
                    Coordinate out_end_corner(out_shape);
                    out_start_corner[0] = static_cast<size_t>(slice_index);
                    out_end_corner[0] = out_start_corner[0] + 1;
                    CoordinateTransform out_transform(
                        out_shape, out_start_corner, out_end_corner, out_strides, out_axis_order);
                    // Create CoordinateTransform for updates slice
                    Coordinate updates_inner_start_corner = *updates_outer_coord_iter;
                    Coordinate updates_inner_end_corner(updates_shape);
                    for (size_t i = 0; i < indices_ndim; i++)
                    {
                        updates_inner_end_corner[i] = updates_inner_start_corner[i] + 1;
                    }
                    CoordinateTransform updates_inner_transform(updates_shape,
                                                                updates_inner_start_corner,
                                                                updates_inner_end_corner,
                                                                updates_strides,
                                                                updates_axis_order);

                    // Add one element from updates to inputs at a time
                    auto updates_inner_coord_iter = updates_inner_transform.begin();
                    for (const Coordinate& out_coord : out_transform)
                    {
                        if (updates_inner_coord_iter == updates_inner_transform.end())
                        {
                            break;
                        }
                        out[out_transform.index(out_coord)] +=
                            updates[updates_inner_transform.index(*updates_inner_coord_iter)];
                        updates_inner_coord_iter++;
                    }
                    updates_outer_coord_iter++;
                }
            }
        }
    }
}
