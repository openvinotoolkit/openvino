// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void sign(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = (arg[i] < T(0) ? T(-1) : (arg[i] > T(0) ? T(1) : T(0)));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
