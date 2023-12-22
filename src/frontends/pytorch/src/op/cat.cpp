// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
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
    FRONT_END_OP_CONVERSION_CHECK(
        list_elems.size() > 1 || !ov::as_type_ptr<v0::Parameter>(first_node),
        "<aten/quantized>::cat is located inside body while inputs are located outside of the body. "
        "This case is not supported.");
    if (list_elems.size() == 1 &&
        !std::dynamic_pointer_cast<op::util::FrameworkNode>(context.get_input(0).get_node_shared_ptr()) && !is_fx) {
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
    auto concat = std::make_shared<v0::Concat>(OutputVector(list_elems.begin(), list_elems.end()), axis);
    return {context.mark_node(concat)};
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
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, context.get_input_size());
    std::deque<Output<Node>> list_elems;
    for (size_t i = 0; i < context.get_input_size() - 1; i++) {
        list_elems.push_back(context.get_input(static_cast<int>(i)));
    }
    auto axis = context.const_input<int64_t>(context.get_input_size() - 1);
    return translate_cat_common(context, list_elems, axis, true);
};

OutputVector translate_quantized_cat(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    const auto&& list_elems = get_list_as_outputs(context.get_input(0));
    auto axis = context.const_input<int64_t>(1);
    FRONT_END_OP_CONVERSION_CHECK(!list_elems.empty(), "Couldn't find quantized input for quantized::cat operation.");
    return {quantize(context,
                     translate_cat_common(context, list_elems, axis, false)[0],
                     context.get_input(2),
                     context.get_input(3),
                     list_elems.front())};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
