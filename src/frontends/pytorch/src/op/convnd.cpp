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

OutputVector translate_convnd(const NodeContext& context) {
    num_inputs_check(context, 7, 7);
    auto strides = context.const_input<Strides>(3);
    // In torch pads at beginning are same as at end
    auto pads = CoordinateDiff(strides.size(), 0);
    auto pad_type = ov::op::PadType::EXPLICIT;
    auto dtype = context.get_input_type(4);
    if (dtype.is<type::Str>()) {
        auto pad_mode = context.const_input<std::string>(4);
        pad_type = convert_pad(pad_mode);
    } else {
        pads = context.const_input<CoordinateDiff>(4);
    }
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<v1::Convolution>(context.get_input(0),
                                                 context.get_input(1),
                                                 strides,
                                                 pads,
                                                 pads,
                                                 dilations,
                                                 pad_type);
    } else {
        conv = std::make_shared<v1::GroupConvolution>(context.get_input(0),
                                                      reshape_kernel_for_group(context, context.get_input(1), groups),
                                                      strides,
                                                      pads,
                                                      pads,
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

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
