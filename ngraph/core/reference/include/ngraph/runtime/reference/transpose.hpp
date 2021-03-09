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
                std::vector<U> range_vector;
                if (axes_order == nullptr)
                {
                    range_vector.resize(arg_size.size());
                    std::iota(range_vector.begin(), range_vector.end(), 0);
                    std::reverse(range_vector.begin(), range_vector.end());
                    axes_order = range_vector.data();
                }

                std::vector<size_t> input_strides(arg_size.size());
                std::vector<size_t> output_strides(arg_size.size());
                input_strides.back() = 1;
                output_strides.back() = 1;

                for (int i = input_strides.size() - 2; i >= 0; i--)
                {
                    input_strides[i] = input_strides[i + 1] * arg_size[i + 1];
                    output_strides[i] = output_strides[i + 1] * arg_size[axes_order[i + 1]];
                }
                for (int i = 0; i < shape_size(arg_size); ++i)
                {
                    size_t in_position = 0;
                    size_t new_position = i;

                    for (int j = 0; j < arg_size.size(); ++j)
                    {
                        in_position +=
                            (new_position / output_strides[j]) * input_strides[axes_order[j]];
                        new_position %= output_strides[j];
                    }
                    out[i] = arg[in_position];
                }
            }
        }
    }
}
