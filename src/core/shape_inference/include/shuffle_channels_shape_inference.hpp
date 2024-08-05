// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/validation_util.hpp"
#include "openvino/op/shuffle_channels.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ShuffleChannels* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);

    const auto& group = op->get_group();
    NODE_VALIDATION_CHECK(op, group >= 1, "The 'group' parameter must be greater or equal to 1.");

    const auto& input_shape = input_shapes[0];
    const auto input_shape_rank = input_shape.rank();

    auto output_shapes = std::vector<TRShape>(1, input_shape);

    if (input_shape_rank.is_static()) {
        NODE_VALIDATION_CHECK(op, input_shape.size() >= 1, "The input tensor's shape is expected to be at least 1D.");
        const auto axis_zb = ov::util::try_normalize_axis(op->get_axis(), input_shape_rank, *op);
        const auto& channel_dim = input_shape[axis_zb];
        NODE_VALIDATION_CHECK(op,
                              channel_dim.is_dynamic() || (channel_dim.get_length() % group) == 0,
                              "The channel dimension size has to be a multiple of the groups parameter value.");

        if (std::is_same<TShape, PartialShape>::value) {
            // overwrite channel dimension to loose symbol
            using TDim = typename TShape::value_type;
            output_shapes.front()[axis_zb] = TDim{channel_dim.get_min_length(), channel_dim.get_max_length()};
        }
    }

    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
