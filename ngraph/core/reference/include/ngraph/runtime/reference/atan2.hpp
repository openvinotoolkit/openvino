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
            template <typename X, typename Y, typename Z>
            void atan2(const X* py, const Y* px, Z* pout, size_t count)
            {
                for (size_t i = 0; i < count; i++)
                {
                    *pout++ = static_cast<Z>(std::atan2(*py++, *px++));
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
