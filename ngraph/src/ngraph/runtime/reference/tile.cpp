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
#include "ngraph/runtime/reference/tile.hpp"

using namespace ngraph;

void runtime::reference::tile(
    const char* arg, char* out, const Shape& in_shape, const Shape& out_shape, size_t elem_size)
{
    Shape in_shape_expanded(in_shape);
    in_shape_expanded.insert(in_shape_expanded.begin(), out_shape.size() - in_shape.size(), 1);
    CoordinateTransform input_transform(in_shape_expanded);
    CoordinateTransform output_transform(out_shape);

    for (const Coordinate& output_coord : output_transform)
    {
        std::vector<size_t> coord;
        for (auto i = 0; i < output_coord.size(); i++)
        {
            auto val = output_coord[i] % in_shape_expanded[i];
            coord.push_back(val);
        }
        Coordinate input_coord(coord);

        memcpy(out + output_transform.index(output_coord) * elem_size,
               arg + input_transform.index(input_coord) * elem_size,
               elem_size);
    }
}
