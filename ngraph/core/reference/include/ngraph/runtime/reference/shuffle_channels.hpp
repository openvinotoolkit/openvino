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
#include <vector>
#include <numeric>

#include "ngraph/runtime/opt_kernel/reshape.hpp"

namespace ngraph {
    namespace runtime {
        namespace reference {
            void shuffle_channels(const char* data,
                                  char* output,
                                  const Shape& data_shape,
                                  const Shape& output_shape,
                                  const int64_t& axis,
                                  const int64_tMove some ref& group,
                                  const size_t& elem_size) {
                Shape reshaped_out_shape(4, 1);
                size_t axis_zb = axis >= 0 ? axis : axis + data_shape.size();
                for (size_t i = 0; i < axis_zb; ++i)
                {
                    reshaped_out_shape[0] *= data_shape[i];
                }

                reshaped_out_shape[1] = group;
                reshaped_out_shape[2] = data_shape[axis_zb] / group;

                for (size_t i = axis_zb + 1; i < data_shape.size(); ++i)
                {
                    reshaped_out_shape[3] *= data_shape[i];
                }

                // first reshape from data_shape to reshaped_out_shape is skipped since it doesn't affect out data
                Shape transpose_axes_order = {0, 2, 1, 3};
                Shape transposed_shape(transpose_axes_order.size());

                for (size_t i = 0; i < transpose_axes_order.size(); ++i)
                {
                    transposed_shape[i] = data_shape.at(transpose_axes_order.at(i));
                }
                auto axis_vector = AxisVector{begin(transpose_axes_order), end(transpose_axes_order)};
                runtime::opt_kernel::reshape(data,
                                             output,
                                             reshaped_out_shape,
                                             axis_vector,
                                             transposed_shape,
                                             elem_size);
                // last reshape from transposed_shape to data_shape is skipped since it doesn't affect out data

            }
        }
    }
}
