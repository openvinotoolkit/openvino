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

#include <cmath>
#include <stdio.h>

#include "ngraph/check.hpp"
#include "ngraph/runtime/reference/slice.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            void slice(const char* arg,
                       char* out,
                       const Shape& arg_shape,
                       const Coordinate& lower_bounds,
                       const Coordinate& upper_bounds,
                       const Strides& strides,
                       const Shape& out_shape,
                       size_t elem_size)
            {
                CoordinateTransform input_transform(arg_shape, lower_bounds, upper_bounds, strides);
                CoordinateTransform output_transform(out_shape);

                CoordinateTransform::Iterator output_it = output_transform.begin();

                NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                             shape_size(output_transform.get_target_shape()));

                for (const Coordinate& in_coord : input_transform)
                {
                    const Coordinate& out_coord = *output_it;

                    memcpy(out + output_transform.index(out_coord) * elem_size,
                           arg + input_transform.index(in_coord) * elem_size,
                           elem_size);

                    ++output_it;
                }
            }
        }
    }
}
