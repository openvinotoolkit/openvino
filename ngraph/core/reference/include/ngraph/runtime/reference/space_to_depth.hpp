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
            void space_to_depth(const char* data,
                                char* output,
                                const Shape& data_shape,
                                const Shape& output_shape,
                                const size_t& block_size,
                                op::SpaceToDepth::SpaceToDepthMode mode,
                                const size_t& elem_size) {
                const size_t n_dim = data_shape.at(0);
                const size_t c_dim = data_shape.at(1);
                const size_t spatial_dim_index = 2;
                const size_t spatial_dims = data_shape.size() - spatial_dim_index;

                for (int i = spatial_dim_index; i < data_shape.size(); ++i)
                {
                    NGRAPH_CHECK(block_size > 0 && data_shape.at(i) % block_size == 0,
                                          "The dimension on position: ",
                                          i,
                                          " equal to: ",
                                          data_shape.at(i),
                                          " must be a multiple of m_blocksize: ",
                                 block_size);
                }

                // First we have to disperse the data from spatial dimensions, then
                // rearrange them so as appropriate chunks of data where close to their
                // destination place. Finally squeeze data from respective dimensions.
                Shape dispersed_shape{n_dim, c_dim};
                for (int i = 0; i < spatial_dims; ++i)
                {
                    dispersed_shape.push_back(data_shape.at(i + spatial_dim_index) / block_size);
                    dispersed_shape.push_back(block_size);
                }
                std::vector<size_t> plain_axes_order(data_shape.size());
                std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);
                std::vector<char> dispersed_data(shape_size(data_shape) * elem_size);
                runtime::opt_kernel::reshape(data,
                                             dispersed_data.data(),
                                             data_shape,
                                             plain_axes_order,
                                             dispersed_shape,
                                             elem_size);
                // calculate axes to transpose
                // [0, 3, 5, ..., spatial_dims + (spatial_dims + 1), 2, 4, ..., K + K])
                std::vector<size_t> axes_order{0};
                for (size_t i = 0, j = 3; i < spatial_dims; ++i, j += 2)
                {
                    axes_order.push_back(j);
                }
                for (size_t i = 0, j = 2; i < spatial_dims; ++i, j += 2)
                {
                    axes_order.push_back(j);
                }

                switch (mode)
                {
                    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ...,
                    // DK/block_size, block_size])
                    // x'' = transpose(x', [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K])
                    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK /
                    // block_size])
                    case op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
                    {
                        axes_order.insert(axes_order.begin() + 1, 1);
                        break;
                    }
                        // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size, ... ,
                        // DK/block_size, block_size])
                        // x'' = transpose(x',  [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K])
                        // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ..., DK /
                        // block_size])
                    case op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
                    default: { axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
                    }
                }
                std::vector<char> transposed_data(shape_size(data_shape) * elem_size);
                Shape post_transpose_shape(axes_order.size());
                for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
                {
                    post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
                }

                runtime::opt_kernel::reshape(dispersed_data.data(),
                                             transposed_data.data(),
                                             dispersed_shape,
                                             axes_order,
                                             post_transpose_shape,
                                             elem_size);

                Shape squeezed_shape{n_dim};
                for (int i = 0; i < spatial_dims; ++i)
                {
                    squeezed_shape.push_back(data_shape.at(spatial_dim_index + i) / block_size);
                }
                squeezed_shape.insert(squeezed_shape.begin() + 1, c_dim * std::pow(block_size, spatial_dims));
                for (size_t i = plain_axes_order.size() - 1; i < post_transpose_shape.size() - 1; ++i)
                {
                    plain_axes_order.push_back(plain_axes_order[i] + 1);
                }
                runtime::opt_kernel::reshape(transposed_data.data(),
                                             output,
                                             post_transpose_shape,
                                             plain_axes_order,
                                             squeezed_shape,
                                             elem_size);
            }
        }
    }
}
