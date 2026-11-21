// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/openvino.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

namespace {

Output<Node> promote_cumprod_input(const NodeContext& context, Output<Node> x) {
    const auto type = get_input_element_type(context, 0);
    if (type == element::boolean || (type.is_static() && type.is_integral_number())) {
        return context.mark_node(std::make_shared<v0::Convert>(x, element::i64));
    }
    return x;
}

Output<Node> cumulative_product(const NodeContext& context, const Output<Node>& x, int64_t dim) {
    const auto rank = x.get_partial_shape().rank();
    if (rank.is_static() && rank.get_length() == 0) {
        return x;
    }

    auto axis_1d = v0::Constant::create(element::i64, Shape{1}, {dim});
    auto zero_scalar = v0::Constant::create(element::i64, Shape{}, {0});
    auto one_1d = v0::Constant::create(element::i64, Shape{1}, {1});
    auto zero_1d = v0::Constant::create(element::i64, Shape{1}, {0});

    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i64));
    auto length = context.mark_node(std::make_shared<v8::Gather>(shape, axis_1d, zero_scalar));
    auto trip_count = v0::Constant::create(element::i64, Shape{}, {-1});
    auto exec_condition = v0::Constant::create(element::boolean, Shape{}, {true});
    auto loop = std::make_shared<v5::Loop>(trip_count, exec_condition);

    auto acc = std::make_shared<v0::Parameter>(x.get_element_type(), x.get_partial_shape());
    auto shift = std::make_shared<v0::Parameter>(element::i64, Shape{1});
    auto body_length = std::make_shared<v0::Parameter>(element::i64, Shape{1});
    auto leading = std::make_shared<v8::Slice>(acc, zero_1d, shift, one_1d, axis_1d);
    auto one = std::make_shared<v1::ConvertLike>(v0::Constant::create(element::f32, Shape{}, {1.0f}), acc);
    auto ones = std::make_shared<v3::Broadcast>(one, std::make_shared<v3::ShapeOf>(leading, element::i64));
    auto head = std::make_shared<v8::Slice>(acc, zero_1d, std::make_shared<v0::Negative>(shift), one_1d, axis_1d);
    auto shifted = std::make_shared<v0::Concat>(OutputVector{ones, head}, dim);
    auto next_acc = std::make_shared<v1::Multiply>(acc, shifted);
    auto next_shift = std::make_shared<v1::Multiply>(shift, v0::Constant::create(element::i64, Shape{1}, {2}));
    auto body_condition = std::make_shared<v0::Squeeze>(std::make_shared<v1::Less>(next_shift, body_length),
                                                        v0::Constant::create(element::i64, Shape{}, {0}));

    auto body = std::make_shared<Model>(OutputVector{next_acc, next_shift, body_condition},
                                        ParameterVector{acc, shift, body_length});
    loop->set_function(body);
    loop->set_special_body_ports({-1, 2});
    loop->set_merged_input(acc, x, next_acc);
    loop->set_merged_input(shift, one_1d, next_shift);
    loop->set_invariant_input(body_length, length);
    auto result = loop->get_iter_value(next_acc, -1);
    return context.mark_node(result.get_node_shared_ptr())->output(result.get_index());
}

}  // namespace

OutputVector translate_cumprod(const NodeContext& context) {
    num_inputs_check(context, 2, 4);
    auto x = context.get_input(0);
    auto dim = context.const_input<int64_t>(1);
    if (!context.input_is_none(2)) {
        x = apply_dtype(context, 2, x);
    } else if (!context.input_is_none(3)) {
        x = context.mark_node(std::make_shared<v1::ConvertLike>(x, context.get_input(3)));
    } else {
        x = promote_cumprod_input(context, x);
    }
    auto result = cumulative_product(context, x, dim);
    if (!context.input_is_none(3)) {
        context.mutate_input(3, result);
    }
    return {result};
};

OutputVector translate_cumprod_fx(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto dim = context.const_input<int64_t>(1);
    if (context.has_attribute("dtype")) {
        auto dtype = context.get_attribute<element::Type>("dtype");
        x = context.mark_node(std::make_shared<v0::Convert>(x, dtype));
    } else {
        x = promote_cumprod_input(context, x);
    }
    return {cumulative_product(context, x, dim)};
};

}  // namespace ov::frontend::pytorch::op
