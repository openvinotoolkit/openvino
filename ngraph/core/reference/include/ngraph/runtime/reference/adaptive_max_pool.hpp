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
            void adaptive_max_pool(const T* arg,
                                   T* out,
                                   int64_t* selected_indices,
                                   const Shape& arg_shape,
                                   const Shape& out_shape)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                size_t spatial_shape_size = 1;
                for (size_t i = 2; i < arg_shape.size(); i++)
                {
                    spatial_shape_size *= arg_shape[i];
                }
                size_t data_shape_size = spatial_shape_size * arg_shape[1];

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate will have the form:
                    //
                    //   (N,chan,i_1,...,i_n)

                    size_t batch_index = out_coord[0];
                    size_t channel = out_coord[1];

                    // Window has the following shape:
                    //
                    //   start = floor(i_2*in_2/out_2, i_3*in_3/out_3,...i_n*in_n/out_n)
                    //   end = ceil((i_2+1)*in_2/out_2, (i_3+1)*in_3/out_3,...(i_n+1)*in_n/out_n)

                    Coordinate input_batch_transform_start(arg_shape.size());
                    Coordinate input_batch_transform_end(arg_shape.size());

                    input_batch_transform_start[0] = batch_index;
                    input_batch_transform_end[0] = batch_index + 1;
                    input_batch_transform_start[1] = channel;
                    input_batch_transform_end[1] = channel + 1;

                    for (size_t i = 2; i < arg_shape.size(); i++)
                    {
                        input_batch_transform_start[i] = (size_t)floor(
                            double(out_coord[i] * arg_shape[i]) / double(out_shape[i]));
                        input_batch_transform_end[i] = (size_t)ceil(
                            double((out_coord[i] + 1) * arg_shape[i]) / double(out_shape[i]));
                    }

                    CoordinateTransform input_batch_transform(
                        arg_shape, input_batch_transform_start, input_batch_transform_end);

                    // As we go, we compute the maximum value:
                    //
                    //   output[O] = max(output[O],arg[I])

                    T result = std::numeric_limits<T>::lowest();
                    int64_t index = -1;

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        auto i = input_batch_transform.index(input_batch_coord);
                        T x = arg[i];
                        if (x > result)
                        {
                            result = x;
                            index = static_cast<int64_t>(i);
                        }
                    }

                    auto out_index = output_transform.index(out_coord);
                    out[out_index] = result;
                    selected_indices[out_index] =
                        index - spatial_shape_size * channel - data_shape_size * batch_index;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
