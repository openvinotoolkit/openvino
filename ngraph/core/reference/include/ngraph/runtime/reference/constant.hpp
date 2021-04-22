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
            void constant(const T* arg0, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    out[i] = arg0[i];
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
