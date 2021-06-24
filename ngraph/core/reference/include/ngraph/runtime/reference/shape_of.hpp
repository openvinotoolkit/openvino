// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/shape.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            inline void shape_of(const Shape& arg_shape, T* out)
            {
                for (size_t i = 0; i < arg_shape.size(); i++)
                {
                    out[i] = static_cast<T>(arg_shape[i]);
                }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
