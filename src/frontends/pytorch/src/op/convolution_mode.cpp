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

OutputVector translate_convolution_mode(NodeContext& context) {
    // Schema: aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[]
    // dilation, int groups) -> Tensor
    auto strides = context.const_input<Strides>(3);
    auto pad_mode = context.const_input<std::string>(4);
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);
    auto pad_const = CoordinateDiff(strides.size(), 0);
    static std::unordered_multimap<std::string, ov::op::PadType> auto_pad_values{{"valid", ov::op::PadType::VALID},
                                                                                 {"same", ov::op::PadType::SAME_UPPER}};

    const auto auto_pad_type_ptr = auto_pad_values.find(pad_mode);
    FRONT_END_OP_CONVERSION_CHECK(auto_pad_type_ptr != auto_pad_values.end(),
                                  "Provided `padding` value: '",
                                  pad_mode,
                                  "' is invalid.");
    auto auto_pad_mode = auto_pad_type_ptr->second;

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = context.mark_node(std::make_shared<opset8::Convolution>(context.get_input(0),
                                                                       context.get_input(1),
                                                                       strides,
                                                                       pad_const,
                                                                       pad_const,
                                                                       dilations,
                                                                       auto_pad_mode));
    } else {
        conv = context.mark_node(std::make_shared<opset8::GroupConvolution>(
            context.get_input(0),
            context.mark_output(reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups)),
            strides,
            pad_const,
            pad_const,
            dilations,
            auto_pad_mode));
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