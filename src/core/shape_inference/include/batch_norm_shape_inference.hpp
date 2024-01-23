// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include <cstdint>

// #include "dimension_util.hpp"
// #include "openvino/core/validation_util.hpp"
#include "openvino/op/batch_norm.hpp"
// #include "openvino/opsets/opset2.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace batch_norm {
namespace util {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> infer_shape(const Op* op,
                                 const TShape& data_input_shape,
                                 const std::vector<TShape>& channel_inputs_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    // NODE_VALIDATION_CHECK(op, input_shapes.size() == 4);

    // Extract channel dimension from input shape.
    auto channel_dim = Dimension::dynamic();
    const auto data_input_rank = data_input_shape.rank();
    if (data_input_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              data_input_rank.get_length() >= 2,
                              "Input argument must have rank of at least 2 (input argument shape: ",
                              data_input_shape,
                              ").");

        channel_dim = data_input_shape[1];
    }

    // Infer gamma/beta/mu/sigma shape, which must be consistent with a vector of size "channel_dim".
    auto channel_shape = PartialShape::dynamic();

    for (const auto& shape : channel_inputs_shapes) {
        NODE_VALIDATION_CHECK(op,
                              PartialShape::merge_into(channel_shape, shape),
                              "Shapes for gamma/beta/mean/variance do not match.");
    }

    NODE_VALIDATION_CHECK(op,
                          channel_shape.merge_rank(1),
                          "Shape for gamma/beta/mean/variance (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(op,
                          Dimension::merge(channel_dim, channel_dim, channel_shape[0]),
                          "Input channel dimension (",
                          channel_dim,
                          ") does not match shape for gamma/beta/mean/variance (",
                          channel_shape,
                          ").");

    NODE_VALIDATION_CHECK(op,
                          channel_dim.is_dynamic() || channel_dim.get_length() >= 1,
                          "Channel count must be at least 1.");

    // Batch result shape is same as the input shape, except we may possibly have inferred more
    // information from the channel count via gamma/beta/etc.
    auto output_shape = data_input_shape;

    if (output_shape.rank().is_static()) {
        output_shape[1] = channel_dim;
    }

    return {std::move(output_shape)};
}
}  // namespace util
}  // namespace batch_norm

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const BatchNormInference* op,
                                 const TShape& data_input_shape,
                                 const std::vector<TShape>& channel_inputs_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return batch_norm::util::infer_shape(op, data_input_shape, channel_inputs_shapes, tensor_accessor);
}
}  // namespace v0

namespace v5 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const BatchNormInference* op,
                                 const TShape& data_input_shape,
                                 const std::vector<TShape>& channel_inputs_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    return batch_norm::util::infer_shape(op, data_input_shape, channel_inputs_shapes, tensor_accessor);
}
}  // namespace v5
}  // namespace op
}  // namespace ov
