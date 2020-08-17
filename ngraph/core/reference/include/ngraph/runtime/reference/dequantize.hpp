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

#include "ngraph/axis_set.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename QUANT, typename REAL>
            void dequantize(const QUANT* input,
                            const REAL* scale,
                            const QUANT* zero_point,
                            REAL* output,
                            const Shape& input_shape,
                            const Shape& scale_zero_point_shape,
                            const AxisSet& axes)
            {
                CoordinateTransform input_transform(input_shape);
                CoordinateTransform scale_zero_point_transform(scale_zero_point_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate scale_zero_point_coord = project(input_coord, axes);

                    output[input_transform.index(input_coord)] =
                        static_cast<REAL>((
                            input[input_transform.index(input_coord)] -
                            zero_point[scale_zero_point_transform.index(scale_zero_point_coord)])) *
                        scale[scale_zero_point_transform.index(scale_zero_point_coord)];
                }
            }
        }
    }
}
