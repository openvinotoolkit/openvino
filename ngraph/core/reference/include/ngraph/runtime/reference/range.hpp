// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
