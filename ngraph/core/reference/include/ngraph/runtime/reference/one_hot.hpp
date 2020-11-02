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
            template <typename INDICES_TYPE, typename OUTPUT_TYPE>
            void one_hot(const INDICES_TYPE* arg,
                         OUTPUT_TYPE* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         size_t one_hot_axis,
                         const OUTPUT_TYPE on_value,
                         const OUTPUT_TYPE off_value)
            {
                // Step 1: Set off_value to the output.
                CoordinateTransform output_transform(out_shape);

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = off_value;
                }

                // Step 2: Write off_value at needed positions, throwing exceptions when invalid
                // conditions are encountered.
                CoordinateTransform input_transform(in_shape);

                for (const Coordinate& input_coord : input_transform)
                {
                    INDICES_TYPE val = arg[input_transform.index(input_coord)];

                    if (std::floor(val) < val || std::floor(val) > val)
                    {
                        continue;
                    }

                    size_t one_hot_pos = static_cast<size_t>(val);

                    if (one_hot_pos >= out_shape[one_hot_axis])
                    {
                        continue;
                    }

                    Coordinate one_hot_coord = inject(input_coord, one_hot_axis, one_hot_pos);

                    out[output_transform.index(one_hot_coord)] = on_value;
                }
            }

            template <typename T>
            void one_hot(const T* arg,
                         T* out,
                         const Shape& in_shape,
                         const Shape& out_shape,
                         size_t one_hot_axis)
            {
                const T on_value = 1;
                const T off_value = 0;
                one_hot<T, T>(arg, out, in_shape, out_shape, one_hot_axis, on_value, off_value);
            }
        }
    }
}
