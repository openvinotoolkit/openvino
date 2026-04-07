// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/paged_causal_conv1d.hpp"
#include "utils.hpp"

namespace ov::op::internal {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const PagedCausalConv1D* op, const std::vector<T>& input_shapes) {
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shapes.size() == 9,
                           "Incorrect number of inputs for PagedCausalConv1D");

    // [batch_size_in_tokens, hidden_size]
    const auto& input_embeds_ps = input_shapes[0];
    // [num_blocks, hidden_size, kernel_size]
    const auto& conv_state_table_ps = input_shapes[1];
    // [out_channels, hidden_size/group_size, conv_kernel_size]
    const auto& conv_weight_ps = input_shapes[2];

    const auto input_hidden_size = input_embeds_ps[1];
    const auto state_hidden_size = conv_state_table_ps[1];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_hidden_size.compatible(state_hidden_size),
                           "The hidden_size dimension of input_embeds and conv_state_table should be compatible, "
                           "but got ",
                           input_hidden_size,
                           " and ",
                           state_hidden_size,
                           ".");

    const auto state_kernel_size = conv_state_table_ps[2];
    const auto weight_kernel_size = conv_weight_ps[2];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           state_kernel_size.compatible(weight_kernel_size),
                           "The kernel_size dimension of conv_state_table and conv_weight should be compatible, "
                           "but got ",
                           state_kernel_size,
                           " and ",
                           weight_kernel_size,
                           ".");

    // output_embeds has the same shape as input_embeds: [batch_size_in_tokens, hidden_size]
    return {input_embeds_ps};
}
}  // namespace ov::op::internal
