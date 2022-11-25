// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

Output<ov::Node> reshape_bias(NodeContext& context, Output<ov::Node> bias, Output<ngraph::Node> conv) {
    auto conv_shape = context.mark_node(std::make_shared<opset8::ShapeOf>(conv));
    auto conv_rank = context.mark_node(std::make_shared<opset8::ShapeOf>(conv_shape));
    auto one_const = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {1}));
    auto two_const = context.mark_node(opset8::Constant::create(element::i64, Shape{1}, {2}));
    auto tail_shape_rank = context.mark_node(std::make_shared<opset8::Subtract>(conv_rank, two_const));
    auto tail_shape = context.mark_node(std::make_shared<opset8::Broadcast>(one_const, tail_shape_rank));
    auto channels_dim = context.mark_node(std::make_shared<opset8::ShapeOf>(bias));
    auto new_shape =
        context.mark_node(std::make_shared<opset8::Concat>(OutputVector{one_const, channels_dim, tail_shape}, 0));

    return context.mark_node(std::make_shared<opset8::Reshape>(bias, new_shape, false));
}

OutputVector translate_convolution(NodeContext& context) {
    // Shchema: aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[]
    // dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool
    // cudnn_enabled, bool allow_tf32) -> Tensor

    auto strides = context.const_input<Strides>(3);
    auto pads = context.const_input<CoordinateDiff>(4);
    auto dilations = context.const_input<Strides>(5);
    bool transposed = context.const_input<bool>(6);
    auto output_padding = context.const_input<CoordinateDiff>(7);
    auto groups = context.const_input<int64_t>(8);

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        if (!transposed) {
            conv = context.mark_node(std::make_shared<opset8::Convolution>(context.get_input(0),
                                                                           context.get_input(1),
                                                                           strides,
                                                                           pads,
                                                                           pads,
                                                                           dilations));
        } else {
            conv = context.mark_node(std::make_shared<opset8::ConvolutionBackpropData>(context.get_input(0),
                                                                                       context.get_input(1),
                                                                                       strides,
                                                                                       pads,
                                                                                       pads,
                                                                                       dilations,
                                                                                       ov::op::PadType::EXPLICIT,
                                                                                       output_padding));
        }
    } else {
        if (!transposed) {
            conv = context.mark_node(std::make_shared<opset8::GroupConvolution>(
                context.get_input(0),
                context.mark_output(
                    reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups)),
                strides,
                pads,
                pads,
                dilations));
        } else {
            conv = context.mark_node(std::make_shared<opset8::GroupConvolutionBackpropData>(
                context.get_input(0),
                context.mark_output(
                    reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups)),
                strides,
                pads,
                pads,
                dilations,
                ov::op::PadType::EXPLICIT,
                output_padding));
        }
    }
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        auto bias_rank = bias.get_partial_shape().rank();
        if (bias_rank == 1) {
            bias = reshape_bias(context, bias, conv);
        }

        conv = context.mark_node(std::make_shared<opset8::Add>(conv, bias));
    }

    return {context.mark_output(conv)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov