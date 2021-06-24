// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void hswish(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = arg[i] * std::min<T>(std::max<T>(arg[i] + 3.0f, 0.0f), 6.0f) / 6.0f;
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
