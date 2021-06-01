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
                if (slope_shape.size() == 1)
                {
                    Shape channel_slope_shape(arg_shape.size(), 1);
                    channel_slope_shape[arg_shape.size() > 1 ? 1 : 0] = slope_shape[0];
                    autobroadcast_binop(arg,
                                        slope,
                                        out,
                                        arg_shape,
                                        channel_slope_shape,
                                        ngraph::op::AutoBroadcastType::NUMPY,
                                        [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
                }
                else
                {
                    autobroadcast_binop(arg,
                                        slope,
                                        out,
                                        arg_shape,
                                        slope_shape,
                                        ngraph::op::AutoBroadcastType::NUMPY,
                                        [](T x, T y) -> T { return x < T(0) ? T(x * y) : x; });
                }

                // int cnt = 0;
                // std::cout << "IN PRELU REF" << std::endl;
                // for (size_t i = 0; i < shape_size(arg_shape); ++i)
                // {
                //     out[i] =
                //         arg[i] < T(0) ? T(arg[i] * slope[cnt++ % shape_size(slope_shape)]) :
                //         arg[i];
                // }
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
