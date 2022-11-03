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

OutputVector translate_avg_pool2d(NodeContext& context) {
    auto kernel = context.const_input<Shape>(1);
    auto strides = context.const_input<Strides>(2);
    auto pads_begin = context.const_input<Shape>(3);  // FIXME: The same 3 is used twice
    auto pads_end = context.const_input<Shape>(3);    // FIXME: The same 3 is used twice
    auto rounding_type = context.const_input<bool>(4) ? ov::op::RoundingType::CEIL : ov::op::RoundingType::FLOOR;
    auto exclude_pad = !context.const_input<bool>(5);
    FRONT_END_OP_CONVERSION_CHECK(context.input_is_none(6),
                                  "Translation for aten::avg_pool2d do not support divisor_override input.");

    return {context.mark_node(std::make_shared<opset8::AvgPool>(context.get_input(0),
                                                                strides,
                                                                pads_begin,
                                                                pads_end,
                                                                kernel,
                                                                exclude_pad,
                                                                rounding_type))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov