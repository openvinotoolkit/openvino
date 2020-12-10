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
#include <type_traits>

#include "ngraph/axis_vector.hpp"
#include "ngraph/check.hpp"
#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            // Return type is `void`, only enabled if `T` is a built-in FP
            // type, or nGraph's `bfloat16` or `float16` type.
            template <typename T>
            typename std::enable_if<std::is_floating_point<T>::value ||
                                    std::is_same<T, bfloat16>::value ||
                                    std::is_same<T, float16>::value>::type
                range(const T* start, const T* step, const size_t& num_elem, T* out)
            {
                for (size_t i = 0; i < num_elem; i++)
                {
                    out[i] = *start + (static_cast<T>(i) * (*step));
                }
            }

            // Return type is `void`, only enabled if `T` is `is_integral`.
            template <typename T>
            typename std::enable_if<std::is_integral<T>::value>::type
                range(const T* start, const T* step, const size_t& num_elem, T* out)
            {
                T val = *start;

                for (size_t i = 0; i < num_elem; i++)
                {
                    out[i] = val;
                    val += *step;
                }
            }
        }
    }
}
