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

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reverse(const T* arg,
                         T* out,
                         const Shape& arg_shape,
                         const Shape& out_shape,
                         const AxisSet& reversed_axes)
            {
                // In fact arg_shape == out_shape, but we'll use both for stylistic consistency with
                // other kernels.
                CoordinateTransform arg_transform(arg_shape);
                CoordinateTransform output_transform(out_shape);

                for (Coordinate out_coord : output_transform)
                {
                    Coordinate arg_coord = out_coord;

                    for (size_t i = 0; i < arg_coord.size(); i++)
                    {
                        if (reversed_axes.count(i) != 0)
                        {
                            arg_coord[i] = arg_shape[i] - arg_coord[i] - 1;
                        }
                    }

                    out[output_transform.index(out_coord)] = arg[arg_transform.index(arg_coord)];
                }
            }
        }
    }
}
