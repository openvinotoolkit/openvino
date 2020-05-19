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
#include "ngraph/runtime/reference/gather_nd.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // Implement gather by calling gather_nd on sub-problems
            // # prepare constant shapes for tensors used for sub problems
            // indices'.shape  = indices.shape[-1] + [1]
            // params'.shape = params.shape[axis:]
            // out'.shape = params'.shape
            // out'.shape[0] = indices.shape[-1]
            // # call sub-problems
            // foreach (params_index, out_index) in outer "axis" dimensions
            //     # params_prime is shared by inner loop
            //     params' = param[params_index] # rank(params') == rank(params) - axis
            //     foreach indices_index in outer N-1 dimensions
            //         indices' = indices[indices_index] # rank(indices') == 2
            //         out_index = out_index + indices_index
            //         out' = out[out_index] # rank(out') == rank(params')
            //         gather_nd(params', indices'', out')
            template <typename T, typename U>
            void gather(const T* params,
                        const U* indices,
                        T* out,
                        const Shape& params_shape,
                        const Shape& indices_shape,
                        const Shape& out_shape,
                        size_t axis)
            {
                using namespace std;
                // prepare shape of params_prime (remove first "axis" dimensions)
                Shape params_prime_shape(params_shape);
                params_prime_shape.erase(params_prime_shape.begin(),
                                         params_prime_shape.begin() + axis);
                // prepare shape of indices_prime
                size_t indices_ndim = static_cast<size_t>(indices_shape.size());
                Shape indices_prime_shape;
                // prepare shape of out_prime (same as params_prime except for first dim)
                Shape out_prime_shape(params_prime_shape);
                if (indices_ndim > 0)
                {
                    out_prime_shape[0] = indices_shape[indices_ndim - 1];
                    indices_prime_shape.emplace_back(indices_shape[indices_ndim - 1]);
                }
                else
                {
                    out_prime_shape[0] = 1;
                }
                indices_prime_shape.emplace_back(1);

                // Create a CoordinateTransform for "out" that visits the outer "axis" dimensions
                size_t out_ndim = static_cast<size_t>(out_shape.size());
                Coordinate out_outer_start_corner(out_ndim, 0);
                Coordinate out_outer_end_corner(out_shape);
                for (size_t i = axis; i < out_ndim; i++)
                {
                    out_outer_end_corner[i] = 1;
                }
                Strides out_outer_strides(out_ndim, 1);
                AxisVector out_outer_axis_order(out_ndim);
                std::iota(out_outer_axis_order.begin(), out_outer_axis_order.end(), 0);
                CoordinateTransform out_outer_transform(out_shape,
                                                        out_outer_start_corner,
                                                        out_outer_end_corner,
                                                        out_outer_strides,
                                                        out_outer_axis_order);

                // Create a CoordinateTransform for "params" that visits the outer "axis" dimensions
                size_t params_ndim = static_cast<size_t>(params_shape.size());
                Coordinate params_outer_start_corner(params_ndim, 0);
                Coordinate params_outer_end_corner(params_shape);
                for (size_t i = axis; i < params_ndim; i++)
                {
                    params_outer_end_corner[i] = 1;
                }
                Strides params_outer_strides(params_ndim, 1);
                AxisVector params_outer_axis_order(params_ndim);
                std::iota(params_outer_axis_order.begin(), params_outer_axis_order.end(), 0);
                CoordinateTransform params_outer_transform(params_shape,
                                                           params_outer_start_corner,
                                                           params_outer_end_corner,
                                                           params_outer_strides,
                                                           params_outer_axis_order);

                // Create a CoordinateTransform for "indices" that visits only the first element
                // along inner most axis
                Coordinate indices_outer_start_corner(indices_ndim, 0);
                Coordinate indices_outer_end_corner(indices_shape);
                if (indices_ndim > 0)
                {
                    indices_outer_end_corner[indices_ndim - 1] = 1;
                }
                Strides indices_outer_strides(indices_ndim, 1);
                AxisVector indices_outer_axis_order(indices_ndim);
                std::iota(indices_outer_axis_order.begin(), indices_outer_axis_order.end(), 0);
                CoordinateTransform indices_outer_transform(indices_shape,
                                                            indices_outer_start_corner,
                                                            indices_outer_end_corner,
                                                            indices_outer_strides,
                                                            indices_outer_axis_order);

                // Create an inner CoordinateTransfrom for "out"
                size_t out_inner_ndim = out_ndim - axis;
                Shape out_inner_shape(out_shape);
                out_inner_shape.erase(out_inner_shape.begin(), out_inner_shape.begin() + axis);
                Coordinate out_inner_start_corner(out_inner_ndim, 0);
                Coordinate out_inner_end_corner(out_inner_shape);
                if (indices_ndim > 0)
                {
                    out_inner_end_corner[indices_ndim - 1] = 1;
                }
                for (size_t i = indices_ndim; i < out_inner_ndim; i++)
                {
                    out_inner_end_corner[i] = 1;
                }
                Strides out_inner_strides(out_inner_ndim, 1);
                AxisVector out_inner_axis_order(out_inner_ndim);
                std::iota(out_inner_axis_order.begin(), out_inner_axis_order.end(), 0);
                CoordinateTransform out_inner_transform(out_inner_shape,
                                                        out_inner_start_corner,
                                                        out_inner_end_corner,
                                                        out_inner_strides,
                                                        out_inner_axis_order);

                auto out_outer_coord_iter = out_outer_transform.begin();
                for (const Coordinate& params_outer_coord : params_outer_transform)
                {
                    const T* params_prime =
                        &params[params_outer_transform.index(params_outer_coord)];
                    T* out_outer = &out[out_outer_transform.index(*out_outer_coord_iter)];

                    auto out_inner_coord_iter = out_inner_transform.begin();
                    for (const Coordinate& indices_outer_coord : indices_outer_transform)
                    {
                        const U* indices_prime =
                            &indices[indices_outer_transform.index(indices_outer_coord)];
                        T* out_prime = &out_outer[out_inner_transform.index(*out_inner_coord_iter)];
                        gather_nd<T, U>(params_prime,
                                        indices_prime,
                                        out_prime,
                                        params_prime_shape,
                                        indices_prime_shape,
                                        out_prime_shape);
                        out_inner_coord_iter++;
                    }
                    out_outer_coord_iter++;
                }
            }
        }
    }
}
