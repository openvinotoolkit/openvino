// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/convolution.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_convolution(const NodeContext& context) {
    // Schema: aten::_convolution(Tensor input, Tensor weight, Tensor? bias, int[] stride, int[] padding, int[]
    // dilation, bool transposed, int[] output_padding, int groups, bool benchmark, bool deterministic, bool
    // cudnn_enabled, bool allow_tf32) -> Tensor

    num_inputs_check(context, 9, 13);
    auto strides = context.const_input<Strides>(3);
    auto pads = context.const_input<CoordinateDiff>(4);
    auto dilations = context.const_input<Strides>(5);
    bool transposed = context.const_input<bool>(6);
    auto output_padding = context.const_input<CoordinateDiff>(7);
    auto groups = context.const_input<int64_t>(8);

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        if (!transposed) {
            conv = context.mark_node(std::make_shared<v1::Convolution>(context.get_input(0),
                                                                       context.get_input(1),
                                                                       strides,
                                                                       pads,
                                                                       pads,
                                                                       dilations));
        } else {
            conv = context.mark_node(std::make_shared<v1::ConvolutionBackpropData>(context.get_input(0),
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
            conv = context.mark_node(
                std::make_shared<v1::GroupConvolution>(context.get_input(0),
                                                       reshape_kernel_for_group(context, context.get_input(1), groups),
                                                       strides,
                                                       pads,
                                                       pads,
                                                       dilations));
        } else {
            conv = context.mark_node(std::make_shared<v1::GroupConvolutionBackpropData>(
                context.get_input(0),
                reshape_kernel_for_group(context, context.get_input(1), groups),
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
            bias = reshape_channelwise(context, bias, conv);
        }

        conv = context.mark_node(std::make_shared<v1::Add>(conv, bias));
    }

    return {context.mark_output(conv)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov