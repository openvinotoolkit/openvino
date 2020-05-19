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
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void argmin(
                const T* arg, U* out, const Shape& in_shape, const Shape& out_shape, size_t axis)
            {
                // take the first elements (i.e. 0 indices) in out_shape - axis as minimums
                memset(out, 0, shape_size(out_shape) * sizeof(U));

                AxisVector av{axis};
                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, av);
                    CoordinateTransform output_transform(out_shape);

                    auto min_index = static_cast<size_t>(out[output_transform.index(output_coord)]);
                    auto min_coord = input_coord;
                    min_coord[axis] = min_index;
                    if (arg[input_transform.index(input_coord)] <
                        arg[input_transform.index(min_coord)])
                    {
                        out[output_transform.index(output_coord)] =
                            static_cast<U>(input_coord[axis]);
                    }
                }
            }
        }
    }
}
