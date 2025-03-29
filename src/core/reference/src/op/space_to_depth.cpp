// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/space_to_depth.hpp"

#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/reference/reshape.hpp"

namespace ov {
namespace reference {
void space_to_depth(const char* const in,
                    const Shape& in_shape,
                    char* const out,
                    const Shape& out_shape,
                    const size_t block_size,
                    const op::v0::SpaceToDepth::SpaceToDepthMode mode,
                    const size_t elem_size) {
    // SpaceToDepth run in tree steps:
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

    for (size_t i = spatial_dim_index; i < in_shape.size(); ++i) {
        OPENVINO_ASSERT(block_size > 0 && in_shape.at(i) % block_size == 0,
                        "SpaceToDepth: The dimension on position: ",
                        i,
                        " equal to: ",
                        in_shape.at(i),
                        " must be a multiple of blocksize: ",
                        block_size);
    }

    Shape dispersed_shape{n_dim, c_dim};
    for (size_t i = 0; i < spatial_dims; ++i) {
        dispersed_shape.push_back(in_shape.at(i + spatial_dim_index) / block_size);
        dispersed_shape.push_back(block_size);
    }

    // calculate axes to transpose
    // [0, 3, 5, ..., spatial_dims + (spatial_dims + 1), 2, 4, ..., K + K])
    std::vector<size_t> axes_order{0};
    for (size_t i = 0, j = 3; i < spatial_dims; ++i, j += 2) {
        axes_order.push_back(j);
    }
    for (size_t i = 0, j = 2; i < spatial_dims; ++i, j += 2) {
        axes_order.push_back(j);
    }

    switch (mode) {
    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size,
    //              ..., DK/block_size, block_size])
    // x'' = transpose(x', [0,  1, 3, 5, ..., K + (K + 1),  2, 4, ..., K + K])
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ...,
    //             DK / block_size])
    case op::v0::SpaceToDepth::SpaceToDepthMode::DEPTH_FIRST: {
        axes_order.insert(axes_order.begin() + 1, 1);
        break;
    }
    // x' = reshape(data, [N, C, D1/block_size, block_size, D2/block_size, block_size,
    //              ... , DK/block_size, block_size])
    // x'' = transpose(x',  [0,  3, 5, ..., K + (K + 1), 1,  2, 4, ..., K + K])
    // y = reshape(x'', [N, C * (block_size ^ K), D1 / block_size, D2 / block_size, ...,
    //             DK / block_size])
    case op::v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST: {
        axes_order.insert(axes_order.begin() + spatial_dims + 1, 1);
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
