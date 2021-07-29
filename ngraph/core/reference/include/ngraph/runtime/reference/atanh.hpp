// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <algorithm>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T,
                      typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
            void atanh(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::atanh(std::min(std::max(arg[i], static_cast<T>(-1.0)), static_cast<T>(1.0f)));
                }
            }

            template <typename T,
                      typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
            void atanh(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::roundl(std::atanh(std::min(std::max(arg[i], static_cast<T>(-1.0)), static_cast<T>(1.0f))));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
