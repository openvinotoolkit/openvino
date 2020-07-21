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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/any.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace
        {
            Shape get_shape_no_keep_dims(const AxisSet& reduction_axes, const Shape& input_shape)
            {
                Shape shape_no_keep_dims;

                for (size_t i = 0; i < input_shape.size(); i++)
                {
                    if (reduction_axes.count(i) == 0)
                    {
                        shape_no_keep_dims.push_back(input_shape[i]);
                    }
                }

                return shape_no_keep_dims;
            }
        }

        namespace reference
        {
            static inline void reduce_logical_and(const char* arg,
                                                  char* out,
                                                  const Shape& input_shape,
                                                  const AxisSet& reduction_axes)
            {
                CoordinateTransform output_transform(
                    get_shape_no_keep_dims(reduction_axes, input_shape));

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 1;
                }

                CoordinateTransform input_transform(input_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes);
                    out[output_transform.index(output_coord)] =
                        out[output_transform.index(output_coord)] &&
                        arg[input_transform.index(input_coord)];
                }
            }

            static inline void reduce_logical_or(const char* arg,
                                                 char* out,
                                                 const Shape& input_shape,
                                                 const AxisSet& reduction_axes)
            {
                runtime::reference::any(arg,
                                        out,
                                        input_shape,
                                        get_shape_no_keep_dims(reduction_axes, input_shape),
                                        reduction_axes);
            }
        }
    }
}
