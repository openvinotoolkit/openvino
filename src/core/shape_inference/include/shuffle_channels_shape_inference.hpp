// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/shuffle_channels.hpp>

#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class TShape>
std::vector<TShape> shape_infer(const ShuffleChannels* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto& group = op->get_group();
    NODE_VALIDATION_CHECK(op, group >= 1, "The 'group' parameter must be greater or equal to 1.");

    const auto& input_shape = input_shapes[0];
    const auto input_shape_rank = input_shape.rank();

    if (input_shape_rank.is_static()) {
        NODE_VALIDATION_CHECK(op, input_shape.size() >= 1, "The input tensor's shape is expected to be at least 1D.");
        const auto axis_zb = static_cast<size_t>(normalize_axis(op, op->get_axis(), input_shape_rank));
        NODE_VALIDATION_CHECK(op,
                              input_shape[axis_zb].is_dynamic() || (input_shape[axis_zb].get_length() % group) == 0,
                              "The channel dimension size has to be a multiple of the groups parameter value.");
    }
    return {input_shape};
}

template <class TShape>
void shape_infer(const ShuffleChannels* op,
                 const std::vector<TShape>& input_shapes,
                 std::vector<TShape>& output_shapes) {
    output_shapes = shape_infer(op, input_shapes);
}

}  // namespace v0
}  // namespace op
}  // namespace ov
