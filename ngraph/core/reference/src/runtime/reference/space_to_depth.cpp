// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/space_to_depth.hpp"
#include <cmath>
#include <numeric>
#include "ngraph/check.hpp"
#include "ngraph/runtime/opt_kernel/reshape.hpp"

namespace ngraph
{
    namespace runtime

    {
        namespace reference
        {
            void space_to_depth(const char* const in,
                                const Shape& in_shape,
                                char* const out,
                                const Shape& out_shape,
                                const size_t block_size,
                                const op::SpaceToDepth::SpaceToDepthMode mode,
                                const size_t elem_size)
            {
                const size_t n_dim = in_shape.at(0);
                const size_t c_dim = in_shape.at(1);
                const size_t spatial_dim_index = 2;
                const size_t spatial_dims = in_shape.size() - spatial_dim_index;

                for (size_t i = spatial_dim_index; i < in_shape.size(); ++i)
                {
                    NGRAPH_CHECK(block_size > 0 && in_shape.at(i) % block_size == 0,
                                 "The dimension on position: ",
                                 i,
                                 " equal to: ",
                                 in_shape.at(i),
                                 " must be a multiple of m_blocksize: ",
                                 block_size);
                }

                // First we have to disperse the data from spatial dimensions, then
                // rearrange them so as appropriate chunks of data where close to their
                // destination place. Finally squeeze data from respective dimensions.
                Shape dispersed_shape{n_dim, c_dim};
                for (size_t i = 0; i < spatial_dims; ++i)
                {
                    dispersed_shape.push_back(in_shape.at(i + spatial_dim_index) / block_size);
                    dispersed_shape.push_back(block_size);
                }
                std::vector<size_t> plain_axes_order(in_shape.size());
                std::iota(plain_axes_order.begin(), plain_axes_order.end(), 0);

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
                // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size,
                // ..., DK/block_size, block_size]) x'' = transpose(x', [0,  1, 3, 5, ..., K + (K +
                // 1),  2, 4, ..., K + K]) y = reshape(x'', [N, C * (block_size ^ K), D1 /
                // block_size, D2 / block_size, ..., DK
                // /
                // block_size])
                case op::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST:
                {
                    axes_order.insert(axes_order.begin() + 1, 1);
                    break;
                }
                // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size,
                // ... , DK/block_size, block_size]) x'' = transpose(x',  [0,  3, 5, ..., K + (K +
                // 1), 1,  2, 4, ..., K + K]) y = reshape(x'', [N, C * (block_size ^ K), D1 /
                // block_size, D2 / block_size, ..., DK
                // /
                // block_size])
                case op::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST:
                default:
                {
                    axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
                }
                }
                std::vector<char> transposed_data(shape_size(in_shape) * elem_size);
                Shape post_transpose_shape(axes_order.size());
                for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx)
                {
                    post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
                }

                runtime::opt_kernel::reshape(
                    in, out, dispersed_shape, axes_order, post_transpose_shape, elem_size);
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
