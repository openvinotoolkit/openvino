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
            void scatter_nd_add(T* inputs,
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
                // Create a CoordinateTransform for "indices" that visits only the first element
                // along inner most axis
                size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                Coordinate indices_outer_start_corner(indices_ndim, 0);
                Coordinate indices_outer_end_corner(indices_shape);
                size_t slice_rank = indices_shape[indices_ndim - 1];
                indices_outer_end_corner[indices_ndim - 1] = 1;
                Strides indices_strides(indices_ndim, 1);
                AxisVector indices_axis_order(indices_ndim);
                std::iota(indices_axis_order.begin(), indices_axis_order.end(), 0);
                CoordinateTransform indices_outer_transform(indices_shape,
                                                            indices_outer_start_corner,
                                                            indices_outer_end_corner,
                                                            indices_strides,
                                                            indices_axis_order);

                // Create a matching CoordinateTransform for "updates" that visits the same outer
                // coordinates
                size_t updates_ndim = static_cast<size_t>(updates_shape.size());
                Strides updates_strides(updates_ndim, 1);
                AxisVector updates_axis_order(updates_ndim);
                std::iota(updates_axis_order.begin(), updates_axis_order.end(), 0);
                Coordinate updates_outer_start_corner(updates_ndim, 0);
                Coordinate updates_outer_end_corner(updates_shape);
                for (size_t i = indices_ndim - 1; i < updates_ndim; i++)
                {
                    updates_outer_end_corner[i] = 1;
                }
                CoordinateTransform updates_outer_transform(updates_shape,
                                                            updates_outer_start_corner,
                                                            updates_outer_end_corner,
                                                            updates_strides,
                                                            updates_axis_order);

                // Add an updates slice to a slice on out indexed by innermost dim ofindices
                size_t out_ndim = static_cast<size_t>(out_shape.size());
                Strides out_strides(out_ndim, 1);
                AxisVector out_axis_order(out_ndim);
                std::iota(out_axis_order.begin(), out_axis_order.end(), 0);

                auto updates_outer_coord_iter = updates_outer_transform.begin();
                for (const Coordinate& indices_coord : indices_outer_transform)
                {
                    if (updates_outer_coord_iter == updates_outer_transform.end())
                    {
                        break;
                    }

                    Coordinate out_start_corner(out_ndim, 0);
                    Coordinate out_end_corner(out_shape);
                    auto indices_index = indices_outer_transform.index(indices_coord);
                    for (size_t i = 0; i < slice_rank; i++)
                    {
                        U index = indices[indices_index];
                        out_start_corner[i] = index;
                        out_end_corner[i] = index + 1;
                        indices_index++;
                    }
                    CoordinateTransform out_transform(
                        out_shape, out_start_corner, out_end_corner, out_strides, out_axis_order);
                    auto updates_index = updates_outer_transform.index(*updates_outer_coord_iter);
                    for (const Coordinate& out_coord : out_transform)
                    {
                        out[out_transform.index(out_coord)] += updates[updates_index];
                        updates_index++;
                    }
                    updates_outer_coord_iter++;
                }
            }
        }
    }
}
