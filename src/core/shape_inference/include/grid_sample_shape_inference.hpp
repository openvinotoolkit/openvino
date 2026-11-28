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

    if (data_shape.rank().is_dynamic() && grid_shape.rank().is_dynamic()) {
        return {PartialShape::dynamic()};
    }
    const size_t out_rank = data_shape.rank().is_static() ? data_shape.size() : grid_shape.size();
    const size_t spatial_rank = out_rank - 2;

    NODE_VALIDATION_CHECK(op,
                          out_rank != 5 || op->get_attributes().mode != GridSample::InterpolationMode::BICUBIC,
                          "GridSample does not support bicubic interpolation for 5D (volumetric) input.");

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
