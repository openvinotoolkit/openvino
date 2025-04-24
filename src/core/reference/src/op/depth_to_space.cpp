// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/depth_to_space.hpp"

#include <cmath>
#include <numeric>

#include "openvino/core/except.hpp"
#include "openvino/reference/reshape.hpp"

namespace ov {
namespace reference {
void depth_to_space(const char* const in,
                    const Shape& in_shape,
                    char* const out,
                    const Shape& out_shape,
                    const size_t block_size,
                    const op::v0::DepthToSpace::DepthToSpaceMode mode,
                    const size_t elem_size) {
    // DepthToSpace run in tree steps:
    // - disperse data from depth channel
    // - rearrange data so as appropriate chunks of data where close to their
    //   destination place
    // - squeeze data from respective dimensions
    //
    // First and third step doesn't change input data in memory, it change only the
    // shape of input. From data layout perspective firs and third step may be
    // omit. The second operation have to be perform on input data with dispared
    // shape (x').
    const size_t n_dim = in_shape.at(0);
    const size_t c_dim = in_shape.at(1);
    const size_t spatial_dim_index = 2;
    const size_t spatial_dims = in_shape.size() - spatial_dim_index;
    const size_t c_dim_divider = static_cast<size_t>(std::pow(block_size, spatial_dims));

    OPENVINO_ASSERT(block_size > 0 && c_dim % c_dim_divider == 0,
                    "DepthToSpace: The input data's 'channels' axis size: ",
                    c_dim,
                    " must be evenly divided by 'block_size'^'spatial_dims': (",
                    c_dim_divider,
                    ", ",
                    block_size,
                    "^",
                    spatial_dims,
                    ")");

    const size_t c_flat = c_dim / c_dim_divider;

    Shape dispersed_shape{n_dim};
    for (size_t i = 0; i < spatial_dims; ++i) {
        dispersed_shape.push_back(block_size);
    }
    for (size_t i = 0; i < spatial_dims; ++i) {
        dispersed_shape.push_back(in_shape.at(spatial_dim_index + i));
    }
    std::vector<size_t> axes_order{0};
    switch (mode) {
    // x' = reshape(data, [N, C / (block_size ^ K), block_size, block_size, ...,
    //              block_size, D1, D2, ..., DK])
    // x'' = transpose(x',
    //                 [0,  1,  K + 2, 2, K + 3, 3, K + 4, 4, ..., K + (K + 1), K + 1])
    // y = reshape(x'',
    //             [N, C / (block_size ^ K), D1 * block_size, D2 * block_size,
    //             D3 * block_size, ..., DK * block_size])
    case op::v0::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST: {
        dispersed_shape.insert(dispersed_shape.begin() + 1, c_flat);
        axes_order.push_back(1);
        for (size_t i = spatial_dim_index; i < in_shape.size(); ++i) {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i);
        }

        break;
    }
    // x' = reshape(data, [N, block_size, block_size, ..., block_size,
    //              C / (block_size ^ K), D1, D2, ..., DK])
    // x'' = transpose(x', [0,  K + 1,  K + 2, 1, K + 3, 2, K + 4, 3, ...,
    //                 K + (K + 1), K])
    // y = reshape(x'', [N, C / (block_size ^ K), D1 * block_size, D2 * block_size,
    //             D3 * block_size, ..., DK * block_size])
    case op::v0::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST: {
        dispersed_shape.insert(dispersed_shape.begin() + spatial_dims + 1, c_flat);
        axes_order.push_back(spatial_dims + 1);
        for (size_t i = 2; i < in_shape.size(); ++i) {
            axes_order.push_back(spatial_dims + i);
            axes_order.push_back(i - 1);
        }
        break;
    }
    }

    Shape post_transpose_shape(axes_order.size());
    for (size_t axis_idx = 0; axis_idx < axes_order.size(); ++axis_idx) {
        post_transpose_shape[axis_idx] = dispersed_shape[axes_order[axis_idx]];
    }

    reshape(in, out, dispersed_shape, axes_order, post_transpose_shape, elem_size);
}

}  // namespace reference
}  // namespace ov
