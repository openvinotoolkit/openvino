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
#include <map>
#include <vector>

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/runtime/reference/sum.hpp"
#include "ngraph/shape_util.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void mean(const T* arg,
                      T* out,
                      const Shape& in_shape,
                      const AxisSet& reduction_axes,
                      bool keep_dims)
            {
                auto out_shape = reduce(in_shape, reduction_axes, keep_dims);
                CoordinateTransform output_transform(out_shape);
                std::vector<T> cs(shape_size(out_shape));

                for (const Coordinate& output_coord : output_transform)
                {
                    out[output_transform.index(output_coord)] = 0;
                    cs[output_transform.index(output_coord)] = 0;
                }

                CoordinateTransform input_transform(in_shape);
                std::map<size_t, int> index_to_count_map;

                for (const Coordinate& input_coord : input_transform)
                {
                    Coordinate output_coord = reduce(input_coord, reduction_axes, keep_dims);

                    T x = arg[input_transform.index(input_coord)];
                    T& z = out[output_transform.index(output_coord)];
                    auto index = output_transform.index(output_coord);
                    if (index_to_count_map.find(index) == index_to_count_map.end())
                    {
                        index_to_count_map[index] = 1;
                    }
                    else
                    {
                        index_to_count_map[index]++;
                    }

                    if (is_finite(x) && is_finite(z))
                    {
                        T& c = cs[output_transform.index(output_coord)];
                        T t = z + (x - c);
                        c = (t - z) - (x - c);
                        z = t;
                    }
                    else
                    {
                        z = z + x;
                    }
                }

                for (const Coordinate& output_coord : output_transform)
                {
                    auto count = index_to_count_map[output_transform.index(output_coord)];
                    out[output_transform.index(output_coord)] =
                        out[output_transform.index(output_coord)] / count;
                }
            }
        }
    }
}
