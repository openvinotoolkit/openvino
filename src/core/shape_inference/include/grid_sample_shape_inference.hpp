// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/validation_util.hpp>
#include <openvino/op/grid_sample.hpp>
#include <vector>

namespace ov {
namespace op {
namespace v9 {

template <class shape_t>
void shape_infer(const GridSample* op, const std::vector<shape_t>& input_shapes, std::vector<shape_t>& output_shapes) {
    NODE_VALIDATION_CHECK(op,
                          input_shapes.size() == 2 && output_shapes.size() == 1,
                          "Incorrect number of input/output shapes in GridSample's shape inference function");
    const auto& data_shape = input_shapes[0];
    NODE_VALIDATION_CHECK(op, data_shape.rank().compatible(4), "The supported shape of the input data tensor is 4D.");
    const auto& grid_shape = input_shapes[1];
    NODE_VALIDATION_CHECK(op, grid_shape.rank().compatible(4), "The supported shape of the grid tensor is 4D.");

    shape_t output_shape;
    output_shape.resize(4);

    auto& batch_dim = output_shape[0];
    auto& channel_dim = output_shape[1];
    auto& H_out_dim = output_shape[2];
    auto& W_out_dim = output_shape[3];

    if (grid_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              grid_shape[3].compatible(2),
                              "The last dimension of grid tensor's shape has to be equal to 2.");
        batch_dim = grid_shape[0];
        H_out_dim = grid_shape[1];
        W_out_dim = grid_shape[2];

        if (data_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                data_shape[0].compatible(grid_shape[0]),
                "The batch dimension in the input data tensor's shape doesn't match the batch dimension in "
                "the grid tensor's shape.");

            // both dimensions should match but use the one which is (possibly) static for the output shape
            if (data_shape[0].is_static()) {
                batch_dim = data_shape[0];
            }

            channel_dim = data_shape[1];
        }
    } else if (data_shape.rank().is_static()) {
        batch_dim = data_shape[0];
        channel_dim = data_shape[1];
    }

    output_shapes[0] = std::move(output_shape);
}

}  // namespace v9
}  // namespace op
}  // namespace ov
