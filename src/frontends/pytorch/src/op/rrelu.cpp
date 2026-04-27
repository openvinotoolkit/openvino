// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/prelu.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector translate_rrelu(const NodeContext& context) {
    // 0 tensor 1 lower 2 upper 3 training
    num_inputs_check(context, 1, 4);
    auto x = context.get_input(0);
    float default_lower = 1 / 8.0f;
    float default_upper = 1 / 3.0f;
    Output<Node> lower = v0::Constant::create(element::f32, Shape{1}, {default_lower});
    Output<Node> upper = v0::Constant::create(element::f32, Shape{1}, {default_upper});
    const auto input_size = context.get_input_size();
    const auto has_lower = (input_size > 1 && !context.input_is_none(1));
    const auto has_upper = (input_size > 2 && !context.input_is_none(2));
    const bool training = (input_size > 3 && !context.input_is_none(3)) ? context.const_input<bool>(3) : false;
    (void)training;
    if (!has_lower && !has_upper) {
        // no limits are given
        auto average = context.mark_node(v0::Constant::create(element::f32, Shape{}, {11 / 48.0f}));
        average = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(average, x));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    } else if (!has_lower && has_upper) {
        upper = context.get_input(2);
        // upper limit is given
        lower = v0::Constant::create(element::f32, Shape{}, {default_lower});
        lower = context.mark_node(std::make_shared<v1::ConvertLike>(lower, x));
        upper = context.mark_node(std::make_shared<v1::ConvertLike>(upper, x));
        auto lower_plus_upper = context.mark_node(std::make_shared<v1::Add>(lower, upper));
        auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0f}));
        two = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(two, x));
        auto average = context.mark_node(std::make_shared<v1::Divide>(lower_plus_upper, two));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    } else if (has_lower && !has_upper) {
        // lower limit is given
        auto lower_limit = context.get_input(1);
        lower_limit = context.mark_node(std::make_shared<v1::ConvertLike>(lower_limit, x));
        auto default_upper_const = context.mark_node(v0::Constant::create(element::f32, Shape{}, {default_upper}));
        default_upper_const = context.mark_node(std::make_shared<v1::ConvertLike>(default_upper_const, x));
        auto lower_plus_upper = context.mark_node(std::make_shared<v1::Add>(lower_limit, default_upper_const));
        auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0f}));
        two = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(two, x));
        auto average = context.mark_node(std::make_shared<v1::Divide>(lower_plus_upper, two));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    } else {
        // both limits are given
        lower = context.get_input(1);
        upper = context.get_input(2);
        lower = context.mark_node(std::make_shared<v1::ConvertLike>(lower, x));
        upper = context.mark_node(std::make_shared<v1::ConvertLike>(upper, x));
        auto lower_plus_upper = context.mark_node(std::make_shared<v1::Add>(lower, upper));
        auto two = context.mark_node(v0::Constant::create(element::f32, Shape{}, {2.0f}));
        two = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(two, x));
        auto average = context.mark_node(std::make_shared<v1::Divide>(lower_plus_upper, two));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    }
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
