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
std::vector<TRShape> infer_shape(const Op* op,
                                 const TShape& data_input_shape,
                                 const std::vector<TShape>& channel_inputs_shapes) {
    // Extract channel dimension from input shape.
    auto channel_dim = TDimension::dynamic();
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
    auto channel_shape = channel_inputs_shapes[0];
    NODE_VALIDATION_CHECK(op,
                          TShape::merge_into(channel_shape, channel_inputs_shapes[1]) &&
                              TShape::merge_into(channel_shape, channel_inputs_shapes[2]) &&
                              TShape::merge_into(channel_shape, channel_inputs_shapes[3]),
                          "Shapes for gamma/beta/mean/variance do not match.");

    NODE_VALIDATION_CHECK(op,
                          channel_shape.merge_rank(1),
                          "Shape for gamma/beta/mean/variance (",
                          channel_shape,
                          ") does not have rank 1.");

    NODE_VALIDATION_CHECK(op,
                          TDimension::merge(channel_dim, channel_dim, channel_shape[0]),
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
    auto outputs_shapes = std::vector<TRShape>{data_input_shape};

    auto& output_shape = outputs_shapes[0];
    if (output_shape.rank().is_static()) {
        output_shape[1] = channel_dim;
    }

    return outputs_shapes;
}
}  // namespace batch_norm

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const v0::BatchNormInference* op, const std::vector<TShape>& inputs_shapes) {
    NODE_VALIDATION_CHECK(op, inputs_shapes.size() == 5);

    static constexpr size_t INPUT_GAMMA = 0;
    static constexpr size_t INPUT_BETA = 1;
    static constexpr size_t INPUT_DATA = 2;
    static constexpr size_t INPUT_MEAN = 3;
    static constexpr size_t INPUT_VARIANCE = 4;
    const auto channel_inputs_shapes = std::vector<TShape>{inputs_shapes[INPUT_GAMMA],
                                                           inputs_shapes[INPUT_BETA],
                                                           inputs_shapes[INPUT_MEAN],
                                                           inputs_shapes[INPUT_VARIANCE]};

    return batch_norm::infer_shape(op, inputs_shapes[INPUT_DATA], channel_inputs_shapes);
}

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const v5::BatchNormInference* op, const std::vector<TShape>& inputs_shapes) {
    NODE_VALIDATION_CHECK(op, inputs_shapes.size() == 5);

    static constexpr size_t INPUT_DATA = 0;
    static constexpr size_t INPUT_GAMMA = 1;
    static constexpr size_t INPUT_BETA = 2;
    static constexpr size_t INPUT_MEAN = 3;
    static constexpr size_t INPUT_VARIANCE = 4;
    const auto channel_inputs_shapes = std::vector<TShape>{inputs_shapes[INPUT_GAMMA],
                                                           inputs_shapes[INPUT_BETA],
                                                           inputs_shapes[INPUT_MEAN],
                                                           inputs_shapes[INPUT_VARIANCE]};
    return batch_norm::infer_shape(op, inputs_shapes[INPUT_DATA], channel_inputs_shapes);
}
}  // namespace op
}  // namespace ov
