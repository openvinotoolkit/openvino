// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/grid_sample.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v9 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GridSample* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 2,
                          "Incorrect number of input shapes in GridSample's shape inference function");
    const auto& data_shape = input_shapes[0];
    const auto& grid_shape = input_shapes[1];

    // GridSample supports 4D (N, C, H, W) and 5D (N, C, D, H, W) data. The grid is expected to have
    // matching rank: 4D grid (N, H_out, W_out, 2) or 5D grid (N, D_out, H_out, W_out, 3). The last grid
    // dimension stores the per-output spatial coordinates and therefore equals the spatial rank (2 or 3).
    NODE_VALIDATION_CHECK(op,
                          data_shape.rank().compatible(4) || data_shape.rank().compatible(5),
                          "The supported shape of the input data tensor is 4D or 5D.");
    NODE_VALIDATION_CHECK(op,
                          grid_shape.rank().compatible(4) || grid_shape.rank().compatible(5),
                          "The supported shape of the grid tensor is 4D or 5D.");

    if (data_shape.rank().is_static() && grid_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              data_shape.size() == grid_shape.size(),
                              "The rank of the data and grid input tensors must be equal.");
    }

    // Determine the output rank: prefer whichever input has a static rank.
    size_t out_rank = 4;
    if (data_shape.rank().is_static()) {
        out_rank = data_shape.size();
    } else if (grid_shape.rank().is_static()) {
        out_rank = grid_shape.size();
    }
    const size_t spatial_rank = out_rank - 2;

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes.front();
    output_shape.resize(out_rank);

    auto& batch_dim = output_shape[0];
    auto& channel_dim = output_shape[1];

    if (grid_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              grid_shape[spatial_rank + 1].compatible(static_cast<int64_t>(spatial_rank)),
                              "The last dimension of the grid tensor's shape has to be equal to the number of "
                              "spatial dimensions of the data tensor.");
        batch_dim = grid_shape[0];
        // Copy the output spatial dimensions (grid[1 .. spatial_rank]) into output[2 .. out_rank).
        for (size_t i = 0; i < spatial_rank; ++i) {
            output_shape[2 + i] = grid_shape[1 + i];
        }

        if (data_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                TShape::value_type::merge(batch_dim, grid_shape[0], data_shape[0]),
                "The batch dimension in the input data tensor's shape doesn't match the batch dimension in "
                "the grid tensor's shape.");
            channel_dim = data_shape[1];
        }
    } else if (data_shape.rank().is_static()) {
        batch_dim = data_shape[0];
        channel_dim = data_shape[1];
    }
    return output_shapes;
}

}  // namespace v9
}  // namespace op
}  // namespace ov
