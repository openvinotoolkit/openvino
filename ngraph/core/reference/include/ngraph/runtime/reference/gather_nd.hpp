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

#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // foreach leaf_vector_index in indices.shape[:-1]
            //     vector = indices[leaf_vector_index]
            //     out[leaf_vector_index:] = params[vector]
            template <typename T, typename U>
            void gather_nd(const T* params,
                           const U* indices,
                           T* out,
                           const Shape& params_shape,
                           const Shape& indices_shape,
                           const Shape& out_shape)
            {
                using namespace std;
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

                // Create a matching CoordinateTransform for "out" that visits the same outer
                // coordinates
                size_t out_ndim = static_cast<size_t>(out_shape.size());
                Coordinate out_start_corner(out_ndim, 0);
                Coordinate out_end_corner(out_shape);
                for (size_t i = indices_ndim - 1; i < out_ndim; i++)
                {
                    out_end_corner[i] = 1;
                }
                Strides out_strides(out_ndim, 1);
                AxisVector out_axis_order(out_ndim);
                std::iota(out_axis_order.begin(), out_axis_order.end(), 0);
                CoordinateTransform out_transform(
                    out_shape, out_start_corner, out_end_corner, out_strides, out_axis_order);
                size_t params_ndim = static_cast<size_t>(params_shape.size());
                Strides params_strides(params_ndim, 1);
                AxisVector params_axis_order(params_ndim);
                std::iota(params_axis_order.begin(), params_axis_order.end(), 0);

                // Gather slices from "params" and copy to "out"
                auto out_coord_iter = out_transform.begin();
                for (const Coordinate& indices_coord : indices_outer_transform)
                {
                    Coordinate params_start_corner(params_ndim, 0);
                    Coordinate params_end_corner(params_shape);
                    auto indices_index = indices_outer_transform.index(indices_coord);
                    for (size_t i = 0; i < slice_rank; i++)
                    {
                        U index = indices[indices_index];
                        // take care of negative indices
                        index = index >= 0 ? index : index + params_shape[i];
                        params_start_corner[i] = index;
                        params_end_corner[i] = index + 1;
                        indices_index++;
                    }
                    CoordinateTransform params_transform(params_shape,
                                                         params_start_corner,
                                                         params_end_corner,
                                                         params_strides,
                                                         params_axis_order);
                    auto out_index = out_transform.index(*out_coord_iter);
                    for (const Coordinate& params_coord : params_transform)
                    {
                        out[out_index] = params[params_transform.index(params_coord)];
                        out_index++;
                    }
                    out_coord_iter++;
                }
            }
        }
    }
}
