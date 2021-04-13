// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <numeric>
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
                if (arg_shape.size() >= 2)
                {
                    NGRAPH_CHECK(arg_shape[1] == shape_size(slope_shape), "If input shape is >= 2, then second second dim of input needs to be equal to slope shape");
                } 
                else {
                    NGRAPH_CHECK(shape_size(slope_shape) == 1, "If input rank < 2, then slop shape has to be 1");
                }
                int cnt = 0;
                int arg_batch_size = arg_shape.size() >= 2 ? std::accumulate(arg_shape.begin() + 1, arg_shape.end(), 0) : 0;
                int arg_spatial_size = 1;
                if (arg_shape.size() < 2)
                {
                    arg_spatial_size = shape_size(arg_shape);
                }
                else if (arg_shape.size() >= 3) 
                {
                    arg_spatial_size = std::accumulate(arg_shape.begin() + 2, arg_shape.end(), 0);
                }
                for (size_t i = 0; i < shape_size(arg_shape); ++i)
                {
                    if (i != 0 && i % arg_spatial_size == 0)  // increment counter after each input channel
                    {
                        cnt++;
                    }

                    if (arg_batch_size != 0 && i % arg_batch_size == 0) // reset counter after each batch
                    {
                        cnt = 0;
                    }
                    out[i] =
                        arg[i] < T(0) ? T(arg[i] * slope[cnt]) : arg[i];
                }
            }
        }
    }
}
