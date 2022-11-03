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

OutputVector translate_conv2d(NodeContext& context) {
    auto strides = context.const_input<Strides>(3);
    auto pads_begin = context.const_input<CoordinateDiff>(4);  // FIXME: The same 4 is used twice
    auto pads_end = context.const_input<CoordinateDiff>(4);    // FIXME: The same 4 is used twice
    auto dilations = context.const_input<Strides>(5);
    auto groups = context.const_input<int64_t>(6);

    std::shared_ptr<ov::Node> conv;
    if (groups == 1) {
        conv = std::make_shared<opset8::Convolution>(context.get_input(0),
                                                     context.get_input(1),
                                                     strides,
                                                     pads_begin,
                                                     pads_end,
                                                     dilations);
    } else {
        conv = std::make_shared<opset8::GroupConvolution>(
            context.get_input(0),
            reshape_kernel_for_group(context, context.get_input(0), context.get_input(1), groups),
            strides,
            pads_begin,
            pads_end,
            dilations);
    }

    return {context.mark_output(make_optional_bias(conv, context, 2, {-2, -1}))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov