// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_convolution_mode(const NodeContext& context) {
    // Schema: aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, int[] stride, str padding, int[]
    // dilation, int groups) -> Tensor
    num_inputs_check(context, 7, 7);
    auto strides = context.const_input<Strides>(3);
    auto pad_mode = context.const_input<std::string>(4);
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);
    const auto config = convert_conv_pad(pad_mode, context.get_input(1), strides, dilations);
    auto data = context.get_input(0);
    if (config.dynamic_same_padding) {
        data = apply_dynamic_same_padding(context, data, context.get_input(1), dilations);
    }

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = context.mark_node(std::make_shared<v1::Convolution>(data,
                                                                   context.get_input(1),
                                                                   strides,
                                                                   config.pads_begin,
                                                                   config.pads_end,
                                                                   dilations,
                                                                   config.pad_type));
    } else {
        conv = context.mark_node(
            std::make_shared<v1::GroupConvolution>(data,
                                                   reshape_kernel_for_group(context, context.get_input(1), groups),
                                                   strides,
                                                   config.pads_begin,
                                                   config.pads_end,
                                                   dilations,
                                                   config.pad_type));
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

}  // namespace ov::frontend::pytorch::op