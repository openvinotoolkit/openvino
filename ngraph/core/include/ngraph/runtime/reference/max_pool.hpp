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

#include <cmath>
#include <numeric>

#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void max_pool_backprop(const T* arg_forward,
                                   const T* delta,
                                   T* out,
                                   const Shape& delta_shape,
                                   const Shape& out_shape, // same as arg_forward_shape
                                   const Shape& window_shape,
                                   const Strides& window_movement_strides,
                                   const Shape& padding_below,
                                   const Shape& padding_above)
            {
                CoordinateTransform out_transform(out_shape);

                for (const Coordinate& out_coord : out_transform)
                {
                    out[out_transform.index(out_coord)] = 0;
                }

                CoordinateTransform delta_transform(delta_shape);

                for (const Coordinate& delta_coord : delta_transform)
                {
                    size_t img_index = delta_coord[0];
                    size_t channel = delta_coord[1];

                    size_t n_image_dimensions = out_shape.size() - 2;
                    Coordinate source_window_transform_start(2 + n_image_dimensions);
                    Coordinate source_window_transform_end(2 + n_image_dimensions);
                    Strides source_window_transform_source_strides(2 + n_image_dimensions, 1);
                    AxisVector source_window_transform_source_axis_order(2 + n_image_dimensions);
                    CoordinateDiff source_window_transform_padding_below(2 + n_image_dimensions);
                    CoordinateDiff source_window_transform_padding_above(2 + n_image_dimensions);

                    source_window_transform_start[0] = img_index;
                    source_window_transform_end[0] = img_index + 1;
                    source_window_transform_start[1] = channel;
                    source_window_transform_end[1] = channel + 1;
                    source_window_transform_padding_below[0] = 0;
                    source_window_transform_padding_below[1] = 0;
                    source_window_transform_padding_above[0] = 0;
                    source_window_transform_padding_above[1] = 0;

                    for (size_t i = 2; i < n_image_dimensions + 2; i++)
                    {
                        size_t window_shape_this_dim = window_shape[i - 2];
                        size_t movement_stride = window_movement_strides[i - 2];

                        source_window_transform_start[i] = movement_stride * delta_coord[i];
                        source_window_transform_end[i] =
                            source_window_transform_start[i] + window_shape_this_dim;
                        source_window_transform_padding_below[i] = padding_below[i - 2];
                        source_window_transform_padding_above[i] = padding_above[i - 2];
                    }
                    std::iota(begin(source_window_transform_source_axis_order),
                              end(source_window_transform_source_axis_order),
                              0);

                    CoordinateTransform source_window_transform(
                        out_shape,
                        source_window_transform_start,
                        source_window_transform_end,
                        source_window_transform_source_strides,
                        source_window_transform_source_axis_order,
                        source_window_transform_padding_below,
                        source_window_transform_padding_above);

                    Coordinate argmax_coord;
                    bool argmax_coord_valid = false;
                    T max_val = 0; // just initializing to keep compiler happy, this 0 is ignored

                    for (const Coordinate& source_window_coord : source_window_transform)
                    {
                        if (source_window_transform.has_source_coordinate(source_window_coord))
                        {
                            T candidate =
                                arg_forward[source_window_transform.index(source_window_coord)];

                            if (!argmax_coord_valid || candidate > max_val)
                            {
                                max_val = candidate;
                                argmax_coord = source_window_coord;
                                argmax_coord_valid = true;
                            }
                        }
                    }

                    if (argmax_coord_valid)
                    {
                        out[source_window_transform.index(argmax_coord)] +=
                            delta[delta_transform.index(delta_coord)];
                    }
                }
            }

            template <typename T>
            void max_pool(const T* arg,
                          T* out,
                          const Shape& arg_shape,
                          const Shape& out_shape,
                          const Shape& window_shape,
                          const Strides& window_movement_strides,
                          const Shape& padding_below,
                          const Shape& padding_above)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate O will have the form:
                    //
                    //   (N,chan,i_1,...,i_n)

                    size_t batch_index = out_coord[0];
                    size_t channel = out_coord[1];

                    // For the input data we need to iterate the coordinate:
                    //
                    //   I:
                    //
                    // over the range (noninclusive on the right):
                    //
                    //   (N,chan,s_1*i_1,s_2*i_2,...,s_n*i_n) ->
                    //
                    //     (N+1,chan+1,s_1*i_1 + window_shape_1,...,s_n*i_n + window_shape_n)
                    //
                    // with unit stride.
                    //
                    // We iterate this over the *padded* data, so below we will need to check for
                    // coordinates that fall in the padding area.

                    size_t n_spatial_dimensions = arg_shape.size() - 2;

                    Coordinate input_batch_transform_start(2 + n_spatial_dimensions);
                    Coordinate input_batch_transform_end(2 + n_spatial_dimensions);
                    Strides input_batch_transform_source_strides(2 + n_spatial_dimensions, 1);
                    AxisVector input_batch_transform_source_axis_order(2 + n_spatial_dimensions);
                    CoordinateDiff input_batch_transform_padding_below(2 + n_spatial_dimensions);
                    CoordinateDiff input_batch_transform_padding_above(2 + n_spatial_dimensions);

                    input_batch_transform_start[0] = batch_index;
                    input_batch_transform_end[0] = batch_index + 1;
                    input_batch_transform_start[1] = channel;
                    input_batch_transform_end[1] = channel + 1;
                    input_batch_transform_padding_below[0] = 0;
                    input_batch_transform_padding_below[1] = 0;
                    input_batch_transform_padding_above[0] = 0;
                    input_batch_transform_padding_above[1] = 0;

                    for (size_t i = 2; i < n_spatial_dimensions + 2; i++)
                    {
                        size_t window_shape_this_dim = window_shape[i - 2];
                        size_t movement_stride = window_movement_strides[i - 2];

                        input_batch_transform_start[i] = movement_stride * out_coord[i];
                        input_batch_transform_end[i] =
                            input_batch_transform_start[i] + window_shape_this_dim;
                        input_batch_transform_padding_below[i] = padding_below[i - 2];
                        input_batch_transform_padding_above[i] = padding_above[i - 2];
                    }

                    for (size_t i = 0; i < arg_shape.size(); i++)
                    {
                        input_batch_transform_source_axis_order[i] = i;
                    }

                    CoordinateTransform input_batch_transform(
                        arg_shape,
                        input_batch_transform_start,
                        input_batch_transform_end,
                        input_batch_transform_source_strides,
                        input_batch_transform_source_axis_order,
                        input_batch_transform_padding_below,
                        input_batch_transform_padding_above);

                    // As we go, we compute the maximum value:
                    //
                    //   output[O] = max(output[O],arg[I])

                    T result = std::numeric_limits<T>::lowest();

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        if (input_batch_transform.has_source_coordinate(input_batch_coord))
                        {
                            T x = arg[input_batch_transform.index(input_batch_coord)];
                            result = x > result ? x : result;
                        }
                    }

                    out[output_transform.index(out_coord)] = result;
                }
            }
        }
    }
}
