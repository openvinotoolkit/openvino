// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_conv_transposend(NodeContext& context) {
    auto strides = context.const_input<Strides>(3);
    // In torch pads at beginning are same as at end
    auto pads = context.const_input<CoordinateDiff>(4);
    auto output_padding = context.const_input<CoordinateDiff>(5);
    auto pad_type = ov::op::PadType::EXPLICIT;
    auto dilations = context.const_input<Strides>(7);
    auto groups = context.const_input<int64_t>(6);
    std::cout << dilations << std::endl;

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<opset10::ConvolutionBackpropData>(context.get_input(0),
                                                                  context.get_input(1),
                                                                  strides,
                                                                  pads,
                                                                  pads,
                                                                  dilations,
                                                                  pad_type,
                                                                  output_padding);
    } else {
        conv = std::make_shared<opset10::GroupConvolutionBackpropData>(
            context.get_input(0),
            reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups),
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
            bias = reshape_conv_bias(context, bias, conv);
        }
        conv = context.mark_node(std::make_shared<opset10::Add>(conv, bias));
    }

    return {conv};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov