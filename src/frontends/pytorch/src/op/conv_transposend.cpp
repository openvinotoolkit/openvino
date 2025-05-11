// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_conv_transposend(const NodeContext& context) {
    num_inputs_check(context, 8, 8);
    auto strides = context.const_input<Strides>(3);
    // PyTorch support only symmetric padding, padding sizes are the same for begins and ends for each dimension
    auto pads = context.const_input<CoordinateDiff>(4);
    auto output_padding = context.const_input<CoordinateDiff>(5);
    auto pad_type = ov::op::PadType::EXPLICIT;
    auto dilations = context.const_input<Strides>(7);
    auto groups = context.const_input<int64_t>(6);
    PYTORCH_OP_CONVERSION_CHECK(groups > 0, "Number of groups for convolution_transpose should be >= 1");

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<v1::ConvolutionBackpropData>(context.get_input(0),
                                                             context.get_input(1),
                                                             strides,
                                                             pads,
                                                             pads,
                                                             dilations,
                                                             pad_type,
                                                             output_padding);
    } else {
        conv = std::make_shared<v1::GroupConvolutionBackpropData>(
            context.get_input(0),
            reshape_kernel_for_group(context, context.get_input(1), groups),
            strides,
            pads,
            pads,
            dilations,
            pad_type,
            output_padding);
    }
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        auto bias_rank = bias.get_partial_shape().rank();
        if (bias_rank == 1) {
            bias = reshape_channelwise(context, bias, conv);
        }
        conv = context.mark_node(std::make_shared<v1::Add>(conv, bias));
    }

    return {conv};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov