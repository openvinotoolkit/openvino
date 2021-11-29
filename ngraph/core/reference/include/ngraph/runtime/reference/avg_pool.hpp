// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void avg_pool_backprop(const T* delta,
                                   T* out,
                                   const Shape& delta_shape,
                                   const Shape& out_shape,
                                   const Shape& window_shape,
                                   const Strides& window_movement_strides,
                                   const Shape& padding_below,
                                   const Shape& padding_above,
                                   bool include_padding_in_avg_computation)
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

                    size_t num_elements_in_window = 0;

                    for (const Coordinate& source_window_coord : source_window_transform)
                    {
                        if (source_window_transform.has_source_coordinate(source_window_coord) ||
                            include_padding_in_avg_computation)
                        {
                            num_elements_in_window++;
                        }
                    }

                    for (const Coordinate& source_window_coord : source_window_transform)
                    {
                        if (source_window_transform.has_source_coordinate(source_window_coord))
                        {
                            size_t out_index = source_window_transform.index(source_window_coord);
                            out[out_index] +=
                                delta[delta_transform.index(delta_coord)] / num_elements_in_window;
                        }
                    }
                }
            }

            template <typename T>
            void avg_pool(const T* arg,
                          T* out,
                          const Shape& arg_shape,
                          const Shape& out_shape,
                          const Shape& window_shape,
                          const Strides& window_movement_strides,
                          const Shape& padding_below,
                          const Shape& padding_above,
                          bool include_padding_in_avg_computation)
            {
                auto old_mode = std::fegetround();
                std::fesetround(FE_TONEAREST);
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
                        // If a window (kernel) is out of arg shape bounds, trim it to fit
                        auto padded_upper_bound =
                            arg_shape[i] + padding_below[i - 2] + padding_above[i - 2];
                        if (input_batch_transform_end[i] > padded_upper_bound)
                        {
                            input_batch_transform_end[i] = padded_upper_bound;
                        }
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

                    // As we go, we compute the sum value:
                    //
                    //   output[O] := output[O] + arg[I]
                    //
                    // and the number of elements:
                    //
                    //   n_elements := n_elements + 1

                    T result = 0;
                    size_t n_elements = 0;

                    // The below conditions are to provide conformance between the ref and plugins:
                    // If exclude_padding is disabled (include_padding... enabled), then:
                    // The size of window doesn't change even if the window was clipped to fit the
                    // input, number of elements will be equal to window_size.width *
                    // window_size.height. The exception from this rule is if padding is not
                    // present, then window size is calculated each time.

                    auto padding_present = padding_below[0] != 0 || padding_below[1] != 0 ||
                                           padding_above[0] != 0 || padding_above[1] != 0;

                    if (include_padding_in_avg_computation && padding_present)
                    {
                        n_elements = shape_size(window_shape);
                    }
                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        bool in_bounds =
                            input_batch_transform.has_source_coordinate(input_batch_coord);

                        if (in_bounds || include_padding_in_avg_computation)
                        {
                            T v = in_bounds ? arg[input_batch_transform.index(input_batch_coord)]
                                            : static_cast<T>(0);
                            result += v;
                            if (!padding_present ||
                                (in_bounds && !include_padding_in_avg_computation))
                            {
                                n_elements++;
                            }
                        }
                    }

                    if (n_elements == 0)
                    {
                        throw std::runtime_error("AvgPool elements == 0, must be non-zero");
                    }

                    if (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value)
                    {
                        out[output_transform.index(out_coord)] =
                            static_cast<T>(std::nearbyint(static_cast<float>(result) / n_elements));
                    }
                    else
                    {
                        out[output_transform.index(out_coord)] = result / n_elements;
                    }
                    std::fesetround(old_mode);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
