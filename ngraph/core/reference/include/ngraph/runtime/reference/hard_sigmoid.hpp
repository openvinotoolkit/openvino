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
            void hard_sigmoid(const T* arg, const T alpha, const T beta, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::max<T>(0.0f, std::min<T>(1.0f, alpha * arg[i] + beta));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
