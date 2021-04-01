//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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
#include <limits>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void max(const T* arg,
                     T* out,
                     const Shape& in_shape,
                     const AxisSet& reduction_axes,
                     bool keep_dims)
            {
                T minval = std::numeric_limits<T>::has_infinity
                               ? T(-std::numeric_limits<T>::infinity())
                               : std::numeric_limits<T>::min();

                auto out_shape = reduce(in_shape, reduction_axes, keep_dims);
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = minval;
                }

                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);

                    T x = arg[input_transform.index(input_coord)];
                    T max = out[output_transform.index(output_coord)];
                    if (x > max)
                    {
                        out[output_transform.index(output_coord)] = x;
                    }
                }
            }
        }
    }
}
