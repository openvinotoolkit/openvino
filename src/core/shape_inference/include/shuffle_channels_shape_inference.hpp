// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/shuffle_channels.hpp>

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ShuffleChannels* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);

    const auto& group = op->get_group();
    NODE_VALIDATION_CHECK(op, group >= 1, "The 'group' parameter must be greater or equal to 1.");

    const auto& input_shape = input_shapes[0];
    const auto input_shape_rank = input_shape.rank();

    if (input_shape_rank.is_static()) {
        const int64_t input_rank_value = static_cast<int64_t>(input_shape.size());
        NODE_VALIDATION_CHECK(op, input_rank_value >= 1, "The input tensor's shape is expected to be at least 1D.");

        const auto& axis = op->get_axis();
        NODE_VALIDATION_CHECK(op,
                              axis < input_rank_value && axis >= (0 - input_rank_value),
                              "The 'axis' parameter for ShuffleChannels has to point to one of the "
                              "input tensor's shape dimensions.");
        size_t axis_zb = static_cast<size_t>(axis >= 0 ? axis : (axis + input_rank_value));

        if (input_shape[axis_zb].is_static()) {
            const auto channel_dim_size = input_shape[axis_zb].get_length();
            NODE_VALIDATION_CHECK(op,
                                  channel_dim_size % group == 0,
                                  "The channel dimension size has to be a multiple of the groups parameter value.");
        }
    }
    output_shapes[0] = input_shape;
}

}  // namespace v0
}  // namespace op
}  // namespace ov