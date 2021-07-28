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
            void abs(const T* arg, T* out, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    // TODO: generic "abs" doesn't work here for some reason.
                    out[i] = (arg[i] < T(0) ? T(-arg[i]) : arg[i]);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
