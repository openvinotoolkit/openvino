// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <array>

#include "openvino/op/group_normalization.hpp"

namespace ov {
namespace op {
namespace v12 {
template <class TShape>
std::vector<TShape> shape_infer(const GroupNormalization* op, const std::vector<TShape>& input_shapes) {
    const auto& inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (inputs_count == 3));

    const auto& data_shape = input_shapes[0];
    const auto& data_rank = data_shape.rank();
    const auto& scale_shape = input_shapes[1];
    const auto& bias_shape = input_shapes[2];

    NODE_VALIDATION_CHECK(op, op->get_num_groups() > 0, "The number of groups needs to be a positive integer value");

    NODE_VALIDATION_CHECK(op, scale_shape.rank().compatible(1), "The scale input is required to be 1D");
    NODE_VALIDATION_CHECK(op, bias_shape.rank().compatible(1), "The bias input is required to be 1D");

    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op, data_rank.get_length() >= 2, "The input tensor is required to be at least 2D");

        const auto& channels_dim = data_shape[1];
        NODE_VALIDATION_CHECK(op,
                              scale_shape.rank().is_dynamic() || channels_dim.compatible(scale_shape[0]),
                              "The scale input shape needs to match the channel dimension in the data input");
        NODE_VALIDATION_CHECK(op,
                              bias_shape.rank().is_dynamic() || channels_dim.compatible(bias_shape[0]),
                              "The bias input shape needs to match the channel dimension in the data input");

        NODE_VALIDATION_CHECK(op,
                              channels_dim.is_dynamic() || op->get_num_groups() <= channels_dim.get_length(),
                              "The number of groups must not exceed the number of channels in the input tensor");

        NODE_VALIDATION_CHECK(op,
                              channels_dim.is_dynamic() || channels_dim.get_length() % op->get_num_groups() == 0,
                              "The number of channels is required to be evenly divisible by the number of groups");
    }

    NODE_VALIDATION_CHECK(op,
                          scale_shape.compatible(bias_shape),
                          "The shapes of both scale and bias inputs need to match");

    return {input_shapes[0]};
}
}  // namespace v12
}  // namespace op
}  // namespace ov
