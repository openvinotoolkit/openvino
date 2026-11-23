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

OutputVector translate_convnd(const NodeContext& context) {
    num_inputs_check(context, 7, 7);
    auto strides = context.const_input<Strides>(3);
    auto dilations = context.const_input<Strides>(5);
    // In torch pads at beginning are same as at end
    auto pads_begin = CoordinateDiff(strides.size(), 0);
    auto pads_end = pads_begin;
    auto pad_type = ov::op::PadType::EXPLICIT;
    bool dynamic_same_padding = false;
    auto dtype = context.get_input_type(4);
    if (dtype.is<type::Str>()) {
        const auto config =
            convert_conv_pad(context.const_input<std::string>(4), context.get_input(1), strides, dilations);
        pad_type = config.pad_type;
        pads_begin = config.pads_begin;
        pads_end = config.pads_end;
        dynamic_same_padding = config.dynamic_same_padding;
    } else {
        pads_begin = context.const_input<CoordinateDiff>(4);
        pads_end = pads_begin;
    }
    auto groups = context.const_input<int64_t>(6);
    auto data = context.get_input(0);
    if (dynamic_same_padding) {
        data = apply_dynamic_same_padding(context, data, context.get_input(1), dilations);
    }

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<v1::Convolution>(data,
                                                 context.get_input(1),
                                                 strides,
                                                 pads_begin,
                                                 pads_end,
                                                 dilations,
                                                 pad_type);
    } else {
        conv = std::make_shared<v1::GroupConvolution>(data,
                                                      reshape_kernel_for_group(context, context.get_input(1), groups),
                                                      strides,
                                                      pads_begin,
                                                      pads_end,
                                                      dilations,
                                                      pad_type);
    }
    conv = context.mark_node(conv);
    if (!context.input_is_none(2)) {
        auto bias = context.get_input(2);
        auto bias_from_visible_context = context.get_input_from_visible_context(2);
        if (ov::as_type_ptr<v0::Constant>(bias_from_visible_context.get_node_shared_ptr())) {
            bias = bias_from_visible_context;
        }
        auto bias_rank = bias.get_partial_shape().rank();
        if (bias_rank == 1) {
            bias = reshape_channelwise(context, bias, conv);
        }
        conv = context.mark_node(std::make_shared<v1::Add>(conv, bias));
    }

    return {conv};
};

}  // namespace ov::frontend::pytorch::op
