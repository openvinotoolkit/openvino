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

#include <cfenv>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "ngraph/axis_vector.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T, typename U>
            void transpose(const T* arg, T* out, Shape arg_size, const U* axes_order = nullptr)
            {
                std::vector<size_t> range_vector;
                if (axes_order == nullptr)
                {
                    range_vector.resize(arg_size.size());
                    std::iota(range_vector.begin(), range_vector.end(), 0);
                    std::reverse(range_vector.begin(), range_vector.end());
                    axes_order = range_vector.data();
                }
                size_t cnt = 0;
                for (size_t i = 0; i < arg_size.size(); ++i)
                {
                    size_t axes = axes_order[i];
                    size_t start = 0;
                    for (size_t j = 0; j < axes; ++j)
                    {
                        start += shape_size(arg_size[j]);
                    }
                    for (size_t j = start; j < start + shape_size(arg_size[axes]); ++j)
                    {
                        out[cnt++] = arg[j];
                    }
                }
            }
        }
    }
}
