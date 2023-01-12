// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_convolution_mode(NodeContext& context) {
    // Schema: aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[]
    // dilation, int groups) -> Tensor
    auto strides = context.const_input<Strides>(3);
    auto pad_mode = context.const_input<std::string>(4);
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);
    auto pad_const = CoordinateDiff(strides.size(), 0);

    const auto auto_pad_type_ptr = TORCH_AUTO_PAD_TO_OV.find(pad_mode);
    FRONT_END_OP_CONVERSION_CHECK(auto_pad_type_ptr != TORCH_AUTO_PAD_TO_OV.end(),
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
            bias = reshape_conv_bias(context, bias, conv);
        }

        conv = context.mark_node(std::make_shared<opset8::Add>(conv, bias));
    }
    return {context.mark_output(conv)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov