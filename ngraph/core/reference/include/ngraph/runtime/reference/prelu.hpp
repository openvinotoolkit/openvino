// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <ngraph/op/util/attr_types.hpp>
#include <ngraph/shape.hpp>

#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void prelu(const T* arg,
                       const T* slope,
                       T* out,
                       const Shape& arg_shape,
                       const Shape& slope_shape)
            {
                int cnt = 0;
                for (size_t i = 0; i < shape_size(arg_shape); ++i)
                {
                    out[i] =
                        arg[i] < T(0) ? T(arg[i] * slope[cnt++ % shape_size(slope_shape)]) : arg[i];
                }
            }
        }
    }
}
