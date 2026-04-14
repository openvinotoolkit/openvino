// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

namespace ov::op::internal {
template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
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
    // conv_bias shape: [out_channels] or [0] (empty = no bias)
    const auto& conv_bias_ps = input_shapes[3];

    const bool embeds_ranked = input_embeds_ps.rank().is_static();
    const bool state_ranked = conv_state_table_ps.rank().is_static();
    const bool weight_ranked = conv_weight_ps.rank().is_static();
    const bool bias_ranked = conv_bias_ps.rank().is_static();

    if (embeds_ranked && state_ranked) {
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
    }

    if (state_ranked && weight_ranked) {
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
    }

    if (embeds_ranked && weight_ranked) {
        // conv_weight[0] is out_channels; spec requires out_channels == hidden_size
        const auto weight_out_channels = conv_weight_ps[0];
        const auto input_hidden_size = input_embeds_ps[1];

        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               weight_out_channels.compatible(input_hidden_size),
                               "The out_channels dimension of conv_weight should be compatible with hidden_size of "
                               "input_embeds, but got ",
                               weight_out_channels,
                               " and ",
                               input_hidden_size,
                               ".");

        if (bias_ranked) {
            const auto bias_size = conv_bias_ps[0];
            const auto zero = ov::Dimension(0);

            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   bias_size.compatible(weight_out_channels) || bias_size.compatible(zero),
                                   "The size of conv_bias should be compatible with out_channels (",
                                   weight_out_channels,
                                   ") or 0 (no bias), but got ",
                                   bias_size,
                                   ".");
        }
    }

    // output_embeds has the same shape as input_embeds: [batch_size_in_tokens, hidden_size]
    return {input_embeds_ps};
}
}  // namespace ov::op::internal
