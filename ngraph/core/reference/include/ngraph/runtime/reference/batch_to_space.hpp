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
#include "ngraph/runtime/reference/strided_slice.hpp"
#include "ngraph/slice_plan.hpp"

namespace ngraph {
    namespace runtime {
        namespace reference {
            void batch_to_space(char* data,
                                const int64_t* block_values,
                                const int64_t* crops_begin_values,
                                const int64_t* crops_end_values,
                                char* output,
                                Shape data_shape,
                                const Shape& block_values_shape,
                                const Shape& crops_begin_shape,
                                const size_t& elem_size
                                ) {
                size_t block_values_size = shape_size(block_values_shape);

                Shape dispersed_shape(1);
                dispersed_shape.insert(dispersed_shape.end(), data_shape.begin(), data_shape.end());
                std::vector<size_t> axes_order(block_values_size + 1);
                std::vector<size_t> plain_axes_order(block_values_size + 1);
                std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
                Shape squeezed_shape(data_shape.begin(), data_shape.end());

                std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);

                Shape post_transpose_shape(axes_order.size());
                std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

                for (size_t block_idx = 1; block_idx < block_values_size; ++block_idx)
                {
                    dispersed_shape[0] = block_values[block_idx];
                    dispersed_shape[1] /= block_values[block_idx];
                    runtime::opt_kernel::reshape(data,
                                                 dispersed_data.data(),
                                                 data_shape,
                                                 plain_axes_order,
                                                 dispersed_shape,
                                                 elem_size);

                    size_t val = 1;
                    for (size_t axis_idx = 0; axis_idx <= block_values_size; ++axis_idx)
                    {
                        if ((block_idx + 1) == axis_idx)
                        {
                            axes_order[axis_idx] = 0;
                        }
                        else
                        {
                            axes_order[axis_idx] = val;
                            val++;
                        }
                    }
                    for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
                    {
                        post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
                    }

                    runtime::opt_kernel::reshape(dispersed_data.data(),
                                                 post_transpose_data.data(),
                                                 dispersed_shape,
                                                 axes_order,
                                                 post_transpose_shape,
                                                 elem_size);
                    squeezed_shape[0] = dispersed_shape[1];
                    squeezed_shape[block_idx] *= block_values[block_idx];
                    dispersed_shape[block_idx + 1] = squeezed_shape[block_idx];
                    runtime::opt_kernel::reshape(post_transpose_data.data(),
                                                 data,
                                                 post_transpose_shape,
                                                 plain_axes_order,
                                                 squeezed_shape,
                                                 elem_size);
                    data_shape = squeezed_shape;
                }

                std::vector<int64_t> upperbounds_values(data_shape.size());
                for (size_t i = 0; i < data_shape.size(); ++i)
                {
                    upperbounds_values[i] = data_shape[i] - crops_end_values[i];
                }

                std::vector<size_t> begin_mask(data_shape.size(), 0);
                std::vector<size_t> end_mask(data_shape.size(), 0);

                std::vector<int64_t> begins(shape_size(crops_begin_shape));
                begins.assign(crops_begin_values, crops_begin_values + shape_size(crops_begin_shape));

                std::vector<int64_t> default_strides(begins.size(), 1);
                SlicePlan slice_plan = make_slice_plan(data_shape,
                                                       begins,
                                                       upperbounds_values,
                                                       default_strides,
                                                       begin_mask,
                                                       end_mask,
                                                       AxisSet(),
                                                       AxisSet(),
                                                       AxisSet());
                runtime::reference::strided_slice(
                        data, output, data_shape, slice_plan, elem_size);
            }
        }
    }
}
