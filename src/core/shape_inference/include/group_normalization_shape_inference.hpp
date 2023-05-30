// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <array>
#include <openvino/core/validation_util.hpp>
#include <openvino/op/group_normalization.hpp>

namespace ov {
namespace op {
namespace v12 {
template <class TShape>
std::vector<TShape> shape_infer(const GroupNormalization* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    const auto& inputs_count = input_shapes.size();
    NODE_VALIDATION_CHECK(op, (inputs_count == 3));
    TShape output_shape;

    const auto data_partial_shape = input_shapes[0];
    const auto data_rank = data_partial_shape.rank();
    const auto scale_partial_shape = input_shapes[1];
    const auto bias_partial_shape = input_shapes[2];

    NODE_VALIDATION_CHECK(op, op->get_num_groups() > 0, "The number of groups needs to be a positive integer value");

    NODE_VALIDATION_CHECK(op,
                          scale_partial_shape.rank().compatible(Dimension{1}),
                          "The scale input is required to be 1D");
    NODE_VALIDATION_CHECK(op,
                          bias_partial_shape.rank().compatible(Dimension{1}),
                          "The bias input is required to be 1D");

    NODE_VALIDATION_CHECK(op,
                          data_rank.is_dynamic() || data_rank.get_length() >= 2,
                          "The input tensor is required to be at least 2D");

    if (data_rank.is_static()) {
        const auto channels_dim = data_partial_shape[1];
        NODE_VALIDATION_CHECK(
            op,
            scale_partial_shape.rank().is_dynamic() || channels_dim.compatible(scale_partial_shape[0]),
            "The scale input shape needs to match the channel dimension in the data input");
        NODE_VALIDATION_CHECK(op,
                              bias_partial_shape.rank().is_dynamic() || channels_dim.compatible(bias_partial_shape[0]),
                              "The bias input shape needs to match the channel dimension in the data input");

        NODE_VALIDATION_CHECK(op,
                              channels_dim.is_dynamic() || op->get_num_groups() <= channels_dim.get_length(),
                              "The number of groups must not exceed the number of channels in the input tensor");

        NODE_VALIDATION_CHECK(op,
                              channels_dim.is_dynamic() || channels_dim.get_length() % op->get_num_groups() == 0,
                              "The number of channels is required to be evenly divisible by the number of groups");
    }

    NODE_VALIDATION_CHECK(op,
                          scale_partial_shape.compatible(bias_partial_shape),
                          "The shapes of both scale and bias inputs need to match");

    return {input_shapes[0]};
}

template <class TShape>
void shape_infer(const GroupNormalization* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes,
                 const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
}
}  // namespace v12
}  // namespace op
}  // namespace ov
