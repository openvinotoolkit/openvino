// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/max_pool.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_max_poolnd(NodeContext& context) {
    num_inputs_check(context, 6, 6);
    auto kernel = context.const_input<Shape>(1);
    auto strides = context.const_input<Strides>(2);
    auto pads = context.const_input<Shape>(3);  // pytorch supports only symmetric paddings
    auto dilations = context.const_input<Strides>(4);
    auto rounding_type = context.const_input<bool>(5) ? RoundingType::CEIL : RoundingType::FLOOR;

    return {context.mark_node(
        std::make_shared<v8::MaxPool>(context.get_input(0), strides, dilations, pads, pads, kernel, rounding_type))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov