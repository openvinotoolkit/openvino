// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
                        // If a window (kernel) is out of arg shape bounds, trim it to fit
                        auto padded_upper_bound =
                            arg_shape[i] + padding_below[i - 2] + padding_above[i - 2];
                        if (input_batch_transform_end[i] > padded_upper_bound)
                        {
                            input_batch_transform_end[i] = padded_upper_bound;
                        }
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
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
