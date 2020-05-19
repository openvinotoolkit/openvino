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
#include <utility>

#include "ngraph/runtime/reference/dot.hpp"
#include "ngraph/runtime/reference/reshape.hpp"
#include "ngraph/shape_util.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void batch_mat_mul(const T* arg0,
                               const T* arg1,
                               T* out,
                               const Shape& arg0_shape,
                               const Shape& arg1_shape,
                               const Shape& out_shape)
            {
                // Call dot for each pair of tensors in the batch
                const size_t batch_size = arg0_shape[0];
                const Shape dot_input0_shape{arg0_shape[1], arg0_shape[2]};
                const Shape dot_input1_shape{arg1_shape[1], arg1_shape[2]};
                const Shape dot_output_shape{out_shape[1], out_shape[2]};
                const size_t input0_offset = shape_size(dot_input0_shape);
                const size_t input1_offset = shape_size(dot_input1_shape);
                const size_t output_offset = shape_size(dot_output_shape);
                for (size_t i = 0; i < batch_size; ++i)
                {
                    dot(arg0 + i * input0_offset,
                        arg1 + i * input1_offset,
                        out + i * output_offset,
                        dot_input0_shape,
                        dot_input1_shape,
                        dot_output_shape,
                        1);
                }
            }
        }
    }
}
