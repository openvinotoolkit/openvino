// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/op/shuffle_channels.hpp>

namespace ov {
namespace op {
namespace v0 {

template <class T>
void shape_infer(const ShuffleChannels* op, const std::vector<T>& input_shapes, std::vector<T>& output_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1 && output_shapes.size() == 1);
    const auto& input_shape = input_shapes[0];
    if (input_shape.is_static()) {
        auto input_rank = input_shape.size();
        NODE_VALIDATION_CHECK(op, input_rank >= 1, "The input tensor's shape is expected to be at least 1D.");

        size_t axis_zb =
            static_cast<size_t>(op->m_axis >= 0 ? op->m_axis : (op->m_axis + static_cast<int64_t>(input_rank)));
        NODE_VALIDATION_CHECK(op,
                              axis_zb < input_rank,
                              "The 'axis' parameter for ShuffleChannels has to point to one of the "
                              "input tensor's shape dimensions.");
        NODE_VALIDATION_CHECK(op, op->m_group >= 1, "The 'group' parameter must be greater or equal to 1.");

        const auto channel_dim_size = input_shape[axis_zb].get_length();
        NODE_VALIDATION_CHECK(op,
                              channel_dim_size % op->m_group == 0,
                              "The channel dimension size has to be a multiple of the groups parameter value.");
    }
    output_shapes[0] = input_shape;
}

}  // namespace v0
}  // namespace op
}  // namespace ov