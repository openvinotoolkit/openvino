// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/op/batch_norm.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace batch_norm {
template <class TShape, class TRShape = result_shape_t<TShape>, class TDimension = typename TShape::value_type>
std::vector<TRShape> infer_shape(const Node* node, const std::vector<TShape>& inputs_shapes) {
    // Extract channel dimension from input shape.
    TDimension channel_dim;
    const auto& data_input_shape = inputs_shapes[0];
    const auto data_input_rank = data_input_shape.rank();
    if (data_input_rank.is_static()) {
        NODE_VALIDATION_CHECK(node,
                              data_input_rank.get_length() >= 2,
                              "Input argument must have rank of at least 2 (input argument shape: ",
                              data_input_shape,
                              ").");

        channel_dim = data_input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size "channel_dim".
    auto channel_shape = inputs_shapes[1];
    NODE_VALIDATION_CHECK(node,
                          TShape::merge_into(channel_shape, inputs_shapes[2]) &&
                              TShape::merge_into(channel_shape, inputs_shapes[3]) &&
                              TShape::merge_into(channel_shape, inputs_shapes[4]),
                          "Shapes for gamma/beta/mean/variance do not match.");

    NODE_VALIDATION_CHECK(node,
                          channel_shape.merge_rank(1),
                          "Shape for gamma/beta/mean/variance (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(node,
                          TDimension::merge(channel_dim, channel_dim, channel_shape[0]),
                          "Input channel dimension (",
                          channel_dim,
                          ") does not match shape for gamma/beta/mean/variance (",
                          channel_shape,
                          ").");

    NODE_VALIDATION_CHECK(node,
                          channel_dim.is_dynamic() || channel_dim.get_length() >= 1,
                          "Channel count must be at least 1.");

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    auto outputs_shapes = std::vector<TRShape>{data_input_shape};

    auto& output_shape = outputs_shapes[0];
    if (data_input_rank.is_static()) {
        output_shape[1] = std::move(channel_dim);
    }

    return outputs_shapes;
}
}  // namespace batch_norm

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const BatchNormInference* op, const std::vector<TShape>& inputs_shapes) {
    NODE_VALIDATION_CHECK(op, inputs_shapes.size() == 5);

    const auto reorder_inputs_shapes =
        std::vector<TShape>{inputs_shapes[2], inputs_shapes[1], inputs_shapes[0], inputs_shapes[3], inputs_shapes[4]};
    return batch_norm::infer_shape(op, reorder_inputs_shapes);
}
}  // namespace v0

namespace v5 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const BatchNormInference* op, const std::vector<TShape>& inputs_shapes) {
    NODE_VALIDATION_CHECK(op, inputs_shapes.size() == 5);

    return batch_norm::infer_shape(op, inputs_shapes);
}
}  // namespace v5
}  // namespace op
}  // namespace ov
