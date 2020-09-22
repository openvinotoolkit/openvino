//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <cmath>
#include <cstddef>

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void reorg_yolo(const T* arg, T* out, const Shape& in_shape, int64_t stride)
            {
                // [N, C, H, W]
                size_t in_N = in_shape[0];
                size_t in_C = in_shape[1];
                size_t in_H = in_shape[2];
                size_t in_W = in_shape[3];

                size_t out_N = in_shape[0];
                size_t out_C = in_shape[1] * (stride * stride);
                size_t out_H = in_shape[2] / stride;
                size_t out_W = in_shape[3] / stride;

                size_t fake_out_C = in_C / (stride * stride);
                if (fake_out_C == 0)
                {
                    throw ngraph_error(
                        "ReorgYolo. For [N, C, H, W] input shape, C >= (stride*stride) is "
                        "required.");
                }
                size_t fake_out_H = in_H * stride;
                size_t fake_out_W = in_W * stride;

                for (size_t n = 0; n < in_N; ++n)
                {
                    for (size_t c = 0; c < in_C; ++c)
                    {
                        for (size_t h = 0; h < in_H; ++h)
                        {
                            for (size_t w = 0; w < in_W; ++w)
                            {
                                size_t dest_index = w + in_W * (h + in_H * (c + in_C * n));

                                size_t new_c = c % fake_out_C;
                                size_t offset = c / fake_out_C;

                                size_t new_w = w * stride + offset % stride;
                                size_t new_h = h * stride + offset / stride;
                                size_t arg_index =
                                    new_w +
                                    fake_out_W * (new_h + fake_out_H * (new_c + fake_out_C * n));

                                out[dest_index] = arg[arg_index];
                            }
                        }
                    }
                }
            }
        }
    }
}
