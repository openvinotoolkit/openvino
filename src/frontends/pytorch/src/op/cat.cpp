// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cat_common(const NodeContext& context, const std::deque<ov::Output<ov::Node>>& list_elems, int64_t axis) {
    if (list_elems.empty()) {
        // couldn't get list elements
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{context.get_input(0)}, 1);
        auto attrs = fw_node->get_attrs();
        // If this fails it means axis is dynamic and aten::cat will be converted to fw node in regular pipeline
        attrs["axis"] = std::to_string(axis);
        fw_node->set_attrs(attrs);
        return {context.mark_node(std::dynamic_pointer_cast<Node>(fw_node))};
    }
    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector(list_elems.begin(), list_elems.end()), axis);
    return {context.mark_node(concat)};
}

OutputVector translate_cat(const NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    return translate_cat_common(context, list_elems, axis);
};

OutputVector translate_cat_fx(const NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, context.get_input_size());
    std::deque<Output<Node>> list_elems;
    list_elems.push_back(context.get_input(0));
    list_elems.push_back(context.get_input(1));

/*    for (size_t i=0; i<context.get_input_size()-1; i++) {
        list_elems.push_back(context.get_input(i));
    }
    auto axis = context.const_input<int64_t>(context.get_input_size()-1); */
     int64_t axis = 0;
    if (!context.input_is_none(2))
        axis = context.const_input<int64_t>(2);

    return translate_cat_common(context, list_elems, axis);
};

OutputVector translate_stack(const NodeContext& context) {
    int64_t axis = 0;
    if (!context.input_is_none(2))
        axis = context.const_input<int64_t>(2);

    auto dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));

    auto stack_input0 = context.mark_node(std::make_shared<v0::Unsqueeze>(context.get_input(0), dim));
    auto stack_input1 = context.mark_node(std::make_shared<v0::Unsqueeze>(context.get_input(1), dim));

    std::deque<Output<Node>> list_elems;
    list_elems.push_back(stack_input0);
    list_elems.push_back(stack_input1);

    return translate_cat_common(context, list_elems, axis);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
