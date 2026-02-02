// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_reverse(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);

    // Detect list reversal. We check if the input is a ListConstruct or if it behaves like a list.
    if (cast_fw_node(data.get_node_shared_ptr(), "prim::ListConstruct") || context.get_input_type(0).is<type::List>()) {
        auto elements = get_list_as_outputs(data);
        std::reverse(elements.begin(), elements.end());
        OutputVector reversed_elements(elements.begin(), elements.end());
        return {context.mark_node(make_list_construct(reversed_elements))};
    }

    // For Tensor inputs, aten::reverse is typically used as a synonym for flip along specified dimensions.
    Output<Node> axes;
    if (context.get_input_size() == 2) {
        axes = context.get_input(1);
    } else {
        // Default to axis 0 if not specified.
        axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    }
    auto reverse_node = context.mark_node(std::make_shared<v1::Reverse>(data, axes, v1::Reverse::Mode::INDEX));
    return {reverse_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
