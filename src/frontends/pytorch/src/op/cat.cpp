// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/convert_promote_types.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_cat_common(const NodeContext& context,
                                  const std::deque<ov::Output<ov::Node>>& list_elems,
                                  int64_t axis,
                                  bool is_fx) {
    if (list_elems.empty()) {
        // couldn't get list elements
        auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{context.get_input(0)}, 1);
        auto attrs = fw_node->get_attrs();
        // If this fails it means axis is dynamic and <aten/quantized>::cat will be converted to fw node in regular
        // pipeline
        attrs["axis"] = std::to_string(axis);
        fw_node->set_attrs(attrs);
        return {context.mark_node(fw_node)};
    }
    auto first_node = list_elems.front().get_node_shared_ptr();
    PYTORCH_OP_CONVERSION_CHECK(
        list_elems.size() > 1 || !ov::as_type_ptr<v0::Parameter>(first_node),
        "<aten/quantized>::cat is located inside body while inputs are located outside of the body. "
        "This case is not supported.");
    if (list_elems.size() == 1 &&
        !ov::as_type_ptr<op::util::FrameworkNode>(context.get_input(0).get_node_shared_ptr()) && !is_fx) {
        // Case when list was merged into tensor. // This case doesn't work with torchfx
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

OutputVector translate_cat(const NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, 3);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    auto out = translate_cat_common(context, list_elems, axis, false);
    if (!context.input_is_none(2)) {
        context.mutate_input(2, out[0]);
    }
    return out;
};

OutputVector translate_cat_fx(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    int64_t axis = 0;
    if (!context.input_is_none(1)) {
        axis = context.const_input<int64_t>(1);
    }
    return translate_cat_common(context, list_elems, axis, true);
};

OutputVector translate_quantized_cat(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    PYTORCH_OP_CONVERSION_CHECK(!list_elems.empty(), "Couldn't find quantized input for quantized::cat operation.");
    return {quantize(context,
                     translate_cat_common(context, list_elems, axis, false)[0],
                     context.get_input(2),
                     context.get_input(3),
                     list_elems.front())};
};

OutputVector translate_stack_fx(const NodeContext& context) {
    num_inputs_check(context, 1, context.get_input_size());
    auto dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    int64_t axis = 0;

    std::deque<Output<Node>> list_elems;
    auto num_elements = context.get_input_size();

    if (!context.get_input_type(num_elements - 1).is<type::List>()) {
        axis = context.const_input<int64_t>(num_elements - 1);
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{}, {axis}));
        num_elements -= 1;
    }

    OutputVector stack_inputs;
    for (size_t i = 0; i < num_elements; i++) {
        stack_inputs.push_back(context.get_input(static_cast<int>(i)));
    }

    // returns the u4 constant if the stack operation is a part of the decompression pattern
    if (const auto& u4_const = u4_compression_stack(stack_inputs, axis))
        return {u4_const};

    for (size_t i = 0; i < num_elements; i++) {
        auto stack_input = context.mark_node(std::make_shared<v0::Unsqueeze>(stack_inputs[i], dim));
        list_elems.push_back(stack_input);
    }
    return translate_cat_common(context, list_elems, axis, true);
}

OutputVector translate_hstack(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    int64_t axis = 1;
    auto out = translate_cat_common(context, list_elems, axis, false);
    if (!context.input_is_none(1)) {
        context.mutate_input(1, out[0]);
    }
    return out;
};

OutputVector translate_vstack(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    int64_t axis = 0;
    auto out = translate_cat_common(context, list_elems, axis, false);
    if (!context.input_is_none(1)) {
        context.mutate_input(1, out[0]);
    }
    return out;
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
