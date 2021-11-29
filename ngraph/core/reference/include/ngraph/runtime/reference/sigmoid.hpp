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
            void sigmoid(const T* arg, T* out, size_t count)
            {
                T exp_value;
                for (size_t i = 0; i < count; i++)
                {
                    exp_value = std::exp(-arg[i]);
                    out[i] = 1 / (1 + exp_value);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
