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
#include "ngraph/runtime/reference/pad.hpp"
#include "ngraph/slice_plan.hpp"

namespace ngraph {
    namespace runtime {
        namespace reference {
            void space_to_batch(char* data,
                                const int64_t* block_values,
                                const int64_t* pads_begin,
                                const int64_t* pads_end,
                                const char* pad_value,
                                char* output,
                                Shape data_shape,
                                const Shape& block_values_shape,
                                const Shape& pads_begin_shape,
                                const Shape& output_shape,
                                const size_t& elem_size
            ) {
                size_t block_values_size = shape_size(block_values_shape);

                const std::vector<char> pad_zero_value(elem_size, 0);

                CoordinateDiff pads_begin_vec(shape_size(pads_begin_shape));
                pads_begin_vec.assign(pads_begin, pads_begin + shape_size(pads_begin_shape));
                CoordinateDiff pads_end_vec(shape_size(pads_begin_shape));
                pads_end_vec.assign(pads_end, pads_end + shape_size(pads_begin_shape));

                Shape padded_shape(data_shape.size());
                for (size_t i = 0; i < data_shape.size(); ++i)
                {
                    padded_shape[i] = data_shape[i] + pads_begin_vec[i] + pads_end_vec[i];
                }

                std::vector<char> padded_data(shape_size(padded_shape) * elem_size);
                ngraph::runtime::reference::pad(data,
                                                pad_value,
                                                padded_data.data(),
                                                elem_size,
                                                data_shape,
                                                padded_shape,
                                                pads_begin_vec,
                                                pads_end_vec,
                                                ngraph::op::PadMode::CONSTANT);
                data_shape = padded_shape;

                Shape dispersed_shape(block_values_size + 1);
                std::vector<size_t> axes_order(block_values_size + 1);
                Shape squeezed_shape(data_shape.begin(), data_shape.end());
                std::vector<size_t> plain_axes_order(block_values_size + 1);
                std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);

                std::vector<char> flat_data(padded_data.begin(), padded_data.end());
                std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
                std::vector<char> post_transpose_data(shape_size(data_shape) * elem_size);

                for (int64_t block_idx = block_values_size - 1; block_idx >= 0; --block_idx)
                {
                    int64_t sq_shape_idx = block_values_size - 1;
                    int64_t axis_idx = axes_order.size() - 1;
                    for (int64_t shape_idx = dispersed_shape.size() - 1; shape_idx >= 0; --shape_idx)
                    {
                        if (shape_idx == (block_idx + 1))
                        {
                            dispersed_shape[shape_idx] = block_values[block_idx];
                            axes_order[0] = shape_idx;
                        }
                        else if (shape_idx == block_idx)
                        {
                            dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx] / block_values[block_idx];
                            axes_order[axis_idx] = shape_idx;
                            axis_idx--;
                            sq_shape_idx--;
                        }
                        else
                        {
                            dispersed_shape[shape_idx] = squeezed_shape[sq_shape_idx];
                            axes_order[axis_idx] = shape_idx;
                            axis_idx--;
                            sq_shape_idx--;
                        }
                    }

                    runtime::opt_kernel::reshape(flat_data.data(),
                                                 dispersed_data.data(),
                                                 data_shape,
                                                 plain_axes_order,
                                                 dispersed_shape,
                                                 elem_size);
                    Shape post_transpose_shape(axes_order.size());
                    for (size_t i = 0; i < axes_order.size(); ++i)
                    {
                        post_transpose_shape[i] = dispersed_shape[axes_order[i]];
                    }

                    runtime::opt_kernel::reshape(dispersed_data.data(),
                                                 post_transpose_data.data(),
                                                 dispersed_shape,
                                                 axes_order,
                                                 post_transpose_shape,
                                                 elem_size);
                    squeezed_shape[0] *= block_values[block_idx];
                    squeezed_shape[block_idx] /= block_values[block_idx];

                    runtime::opt_kernel::reshape(post_transpose_data.data(),
                                                 flat_data.data(),
                                                 post_transpose_shape,
                                                 plain_axes_order,
                                                 squeezed_shape,
                                                 elem_size);
                    data_shape = squeezed_shape;
                }
                memcpy(output, flat_data.data(), elem_size * shape_size(output_shape));
            }
        }
    }
}
