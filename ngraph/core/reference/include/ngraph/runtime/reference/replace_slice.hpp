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

#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void replace_slice(const T* arg0, // replacement context
                               const T* arg1, // replacement value
                               T* out,
                               const Shape& arg1_shape,
                               const Coordinate& lower_bounds,
                               const Coordinate& upper_bounds,
                               const Strides& strides,
                               const Shape& out_shape)
            {
                // Step 1: Copy the entire replacement context to the output.
                CoordinateTransform copy_transform(out_shape);

                for (Coordinate copy_coord : copy_transform)
                {
                    out[copy_transform.index(copy_coord)] = arg0[copy_transform.index(copy_coord)];
                }

                // Step 2: Overwrite the slice for replacement.
                CoordinateTransform input_transform(arg1_shape);
                CoordinateTransform output_transform(
                    out_shape, lower_bounds, upper_bounds, strides);

                NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                             shape_size(output_transform.get_target_shape()));

                CoordinateTransform::Iterator output_it = output_transform.begin();

                for (const Coordinate& input_coord : input_transform)
                {
                    if (output_it == output_transform.end())
                        break;
                    const Coordinate& output_coord = *output_it;

                    out[output_transform.index(output_coord)] =
                        arg1[input_transform.index(input_coord)];

                    ++output_it;
                }
            }
        }
    }
}
