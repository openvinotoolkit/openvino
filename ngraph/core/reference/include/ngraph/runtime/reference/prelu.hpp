// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <cassert>
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
                // assert(arg_shape.size() > 2);
                // size_t batch_size = arg_shape[0] * arg_shape[1];
                // size_t sptial_size = shape_size(arg_shape) / batch_size;
                // for (int n = 0; n < batch_size; ++n)
                // {
                //     T slope_n = shape_size(slope_shape) == 1 ? slope[0] : slope[n % shape_size(slope_shape)];
                //     for (int sp = 0; sp < sptial_size; ++sp)
                //     {
                //         size_t inx = n * sptial_size + sp;
                //         out[inx] =
                //             arg[inx] < T(0) ? T(arg[inx] * slope_n) : arg[inx];
                //     }
                // }
            }
        }
    }
}
