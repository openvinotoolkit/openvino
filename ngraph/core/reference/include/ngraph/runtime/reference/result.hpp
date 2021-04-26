// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void result(const T* arg, T* out, size_t count)
            {
                memcpy(out, arg, sizeof(T) * count);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
