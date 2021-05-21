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
            void cos(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = std::cos(arg[i]);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
