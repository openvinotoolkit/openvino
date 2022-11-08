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

OutputVector translate_max_pool2d(NodeContext& context) {
    auto kernel = context.const_input<Shape>(1);
    auto strides = context.const_input<Strides>(2);
    auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
    auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
    auto dilations = context.const_input<Strides>(4);
    auto rounding_type = context.const_input<bool>(5) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;

    return {context.mark_node(std::make_shared<opset8::MaxPool>(context.get_input(0),
                                                                strides,
                                                                dilations,
                                                                pads_begin,
                                                                pads_end,
                                                                kernel,
                                                                rounding_type))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov