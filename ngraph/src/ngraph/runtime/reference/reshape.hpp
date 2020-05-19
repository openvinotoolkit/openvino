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

#include "ngraph/axis_vector.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reshape(const T* arg,
                         T* out,
                         const Shape& in_shape,
                         const AxisVector& in_axis_order,
                         const Shape& out_shape)
            {
                // Unfortunately we don't yet have a constructor for CoordinateTransform that lets
                // us pass only source_space_shape
                // and source_axis_order so we have to construct the defaults here.
                Shape in_start_corner(in_shape.size(), 0); // (0,...0)
                Strides in_strides(in_shape.size(), 1);    // (1,...,1)

                CoordinateTransform input_transform(
                    in_shape, in_start_corner, in_shape, in_strides, in_axis_order);
                CoordinateTransform output_transform(out_shape);

                NGRAPH_CHECK(shape_size(input_transform.get_target_shape()) ==
                             shape_size(output_transform.get_target_shape()));

                CoordinateTransform::Iterator output_it = output_transform.begin();

                for (const Coordinate& input_coord : input_transform)
                {
                    const Coordinate& output_coord = *output_it;

                    out[output_transform.index(output_coord)] =
                        arg[input_transform.index(input_coord)];

                    ++output_it;
                }
            }
        }
    }
}
