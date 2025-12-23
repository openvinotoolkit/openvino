// // Copyright (C) 2018-2025 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

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
    num_inputs_check(context, 1, 4);
    auto x = context.get_input(0);
    float default_lower = 1 / 8.0f;
    float default_upper = 1 / 3.0f;
    float default_mean = 11 / 48.0f;
    Output<Node> lower = v0::Constant::create(element::f32, Shape{1}, {default_lower});
    Output<Node> upper = v0::Constant::create(element::f32, Shape{1}, {default_upper});
    if (context.input_is_none(1) && context.input_is_none(2)) {
        // no limits are given
        auto average = context.mark_node(v0::Constant::create(element::f32, Shape{}, {default_mean}));
        average = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(average, x));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    } else if (context.input_is_none(1) && !context.input_is_none(2)) {
        // upper limit is given
        float upper_limit = context.const_input<float>(2);
        float mean = default_mean + (upper_limit - default_upper) / 2.0f;
        auto average = context.mark_node(v0::Constant::create(element::f32, Shape{}, {mean}));
        average = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(average, x));
        return {context.mark_node(std::make_shared<v0::PRelu>(x, average))};
    } else if (!context.input_is_none(1) && context.input_is_none(2)) {
        // lower limit is given
        float lower_limit = context.const_input<float>(1);
        float mean = default_mean + (lower_limit - default_lower) / 2.0f;
        auto average = context.mark_node(v0::Constant::create(element::f32, Shape{}, {mean}));
        average = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(average, x));
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
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov