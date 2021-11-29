// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

#include "ngraph/type/bfloat16.hpp"
#include "ngraph/type/float16.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T,
                      typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
            void erf(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::erf(arg[i]);
                }
            }

            template <typename T,
                      typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
            void erf(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::round(std::erf(arg[i]));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
