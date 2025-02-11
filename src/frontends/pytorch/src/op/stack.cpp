// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_stack_common(const NodeContext& context,
                                    const std::deque<ov::Output<ov::Node>>& list_elems, 
                                    int64_t axis) {
    auto first_node = list_elems.front().get_node_shared_ptr();
    if (list_elems.size() == 1 &&
        !ov::as_type_ptr<op::util::FrameworkNode>(context.get_input(0).get_node_shared_ptr())) {
        auto tensor = list_elems[0];
        auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(tensor, element::i32));
        auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
        auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
        auto axis_const = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {axis}));
        auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
        auto int_max =
            context.mark_node(v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>().max()}));
        auto shape_sliced = context.mark_node(std::make_shared<v8::Slice>(shape, one, int_max, one));
        auto new_shape =
            context.mark_node(std::make_shared<v12::ScatterElementsUpdate>(shape_sliced, axis_const, neg_1, zero));
        return {context.mark_node(std::make_shared<v1::Reshape>(tensor, new_shape, false))};
    }

    const auto first_in_type = list_elems.front().get_element_type();
    const bool is_mixed_type =
        list_elems.size() > 1 && (std::any_of(std::next(list_elems.begin()),
                                              list_elems.end(),
                                              [&first_in_type](const ov::Output<ov::Node>& input) {
                                                  return input.get_element_type() != first_in_type ||
                                                         input.get_element_type() == ov::element::dynamic;
                                              }));
    auto inputs_vec = OutputVector(list_elems.begin(), list_elems.end());
    if (is_mixed_type) {
        auto node_of_type = inputs_vec[0];
        for (size_t i = 1; i < inputs_vec.size(); ++i) {
            auto cpt = context.mark_node(std::make_shared<v14::ConvertPromoteTypes>(node_of_type, list_elems[i], true));
            node_of_type = cpt->output(0);
            inputs_vec[i] = cpt->output(1);
        }

        inputs_vec[0] = node_of_type;
        const auto unified_type = node_of_type.get_element_type();
        for (size_t i = 1; i < inputs_vec.size(); ++i) {
            if (inputs_vec[i].get_element_type() != unified_type ||
                inputs_vec[i].get_element_type() == ov::element::dynamic) {
                inputs_vec[i] = context.mark_node(std::make_shared<v1::ConvertLike>(list_elems[i], node_of_type));
            }
        }
        auto concat = std::make_shared<v0::Concat>(inputs_vec, axis);
        return {context.mark_node(concat)};
    }

    return {context.mark_node(std::make_shared<v0::Concat>(inputs_vec, axis))};
}

OutputVector translate_hstack(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    int64_t axis = 1;
    auto out = translate_stack_common(context, list_elems, axis);
    if (!context.input_is_none(2)) {
        context.mutate_input(1, out[0]);
    }
    return out;
};

OutputVector translate_vstack(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    int64_t axis = 0;
    auto out = translate_stack_common(context, list_elems, axis);
    if (!context.input_is_none(2)) {
        context.mutate_input(1, out[0]);
    }
    return out;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov