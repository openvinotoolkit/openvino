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

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void roll(const T* arg,
                      const int64_t* shift,
                      const int64_t* axes,
                      T* out,
                      const Shape& arg_shape,
                      const Shape& shift_shape,
                      const Shape& axes_shape)
            {
                std::vector<int64_t> axes_vector = std::vector<int64_t>(axes, axes + axes_shape[0]);
                for (auto& axis : axes_vector)
                {
                    if (axis < 0)
                        axis += arg_shape.size();
                }

                CoordinateTransform arg_transform{arg_shape};
                for (const Coordinate& arg_coord : arg_transform)
                {
                    Coordinate new_coord = arg_coord;
                    for (size_t i = 0; i < shift_shape[0]; ++i)
                    {
                        int64_t axis = axes_vector[i];
                        int64_t new_coord_value = (int64_t)new_coord[axis] + shift[i];
                        int64_t dim_size = arg_shape[axis];
                        // the modulo which supports negative values
                        new_coord[axis] = (new_coord_value % dim_size + dim_size) % dim_size;
                    }
                    out[arg_transform.index(new_coord)] = arg[arg_transform.index(arg_coord)];
                }
            }
        }
    }
}
