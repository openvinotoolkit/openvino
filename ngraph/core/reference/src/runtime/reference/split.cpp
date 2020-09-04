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
#include "ngraph/runtime/reference/split.hpp"

using namespace ngraph;

void runtime::reference::split(const char* data,
                               const Shape& data_shape,
                               size_t elem_size,
                               int64_t axis,
                               size_t num_splits,
                               char** out_data)
{
    const size_t part_length = data_shape.at(axis) / num_splits;

    Shape output_shape = data_shape;
    output_shape.at(axis) = part_length;

    std::vector<size_t> lower_bounds(data_shape.size(), 0);
    std::vector<size_t> upper_bounds = data_shape;
    upper_bounds.at(axis) = part_length;

    for (size_t i = 0; i < num_splits; ++i)
    {
        runtime::reference::slice(data,
                                  out_data[i],
                                  data_shape,
                                  lower_bounds,
                                  upper_bounds,
                                  Strides(lower_bounds.size(), 1),
                                  output_shape,
                                  elem_size);
        lower_bounds.at(axis) += part_length;
        upper_bounds.at(axis) += part_length;
    }
}
