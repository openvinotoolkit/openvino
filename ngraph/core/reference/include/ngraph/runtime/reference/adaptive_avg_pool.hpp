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
            void adaptive_avg_pool(const T* arg,
                                   T* out,
                                   const Shape& arg_shape,
                                   const Shape& out_shape)
            {
                // At the outermost level we will walk over every output coordinate O.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& out_coord : output_transform)
                {
                    // Our output coordinate will have the form:
                    //
                    //   (N,chan,i_2,...,i_n)

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

                    // As we go, we compute the sum value:
                    //
                    //   output[O] := output[O] + arg[I]
                    //
                    // and the number of elements:
                    //
                    //   n_elements := n_elements + 1

                    T result = 0;
                    size_t n_elements = 0;

                    for (const Coordinate& input_batch_coord : input_batch_transform)
                    {
                        result += arg[input_batch_transform.index(input_batch_coord)];
                        n_elements++;
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
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
