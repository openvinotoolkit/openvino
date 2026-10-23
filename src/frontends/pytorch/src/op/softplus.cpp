// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softplus.hpp"

#include <cmath>

#include "openvino/core/validation_util.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {
using namespace ov::op;

OutputVector translate_softplus(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto input = context.get_input(0);
    auto original_type = input.get_element_type();
    if (original_type == element::bf16 && (context.input_is_none(1) || context.const_input<float>(1) == 1.0f) &&
        (context.input_is_none(2) || context.const_input<float>(2) == 20.0f) && !context.has_attribute("beta") &&
        !context.has_attribute("threshold")) {
        if (auto constant = ov::util::get_constant_from_source(input)) {
            auto values = constant->cast_vector<float>();
            for (auto& value : values) {
                value = value > 20 ? value : std::log1p(std::exp(value));
            }
            return {context.mark_node(make_bfloat16_constant(constant->get_shape(), values))};
        }
    }
    if (original_type == element::bf16 || original_type == element::f16) {
        input = context.mark_node(std::make_shared<v0::Convert>(input, element::f32));
    }
    auto scalar = [&](float value) -> Output<Node> {
        auto constant = context.mark_node(v0::Constant::create(element::f32, Shape{}, {value}));
        return context.mark_node(std::make_shared<v1::ConvertLike>(constant, input));
    };
    auto beta = context.has_attribute("beta") ? context.get_attribute<Output<Node>>("beta")
                                              : (context.input_is_none(1) ? scalar(1) : context.get_input(1));
    beta = context.mark_node(std::make_shared<v1::ConvertLike>(beta, input));
    auto threshold = context.has_attribute("threshold")
                         ? context.get_attribute<Output<Node>>("threshold")
                         : (context.input_is_none(2) ? scalar(20) : context.get_input(2));
    threshold = context.mark_node(std::make_shared<v1::ConvertLike>(threshold, input));
    auto scaled = context.mark_node(std::make_shared<v1::Multiply>(input, beta));
    auto boundary = scalar(-4);
    auto bounded = context.mark_node(std::make_shared<v1::Minimum>(scaled, boundary));
    auto exp = context.mark_node(std::make_shared<v0::Exp>(bounded));
    // log1p(t) avoids cancellation in the negative tail; t <= exp(-4).
    Output<Node> polynomial = scalar(-0.25f);
    for (float coefficient : {1.0f / 3, -0.5f, 1.0f}) {
        auto product = context.mark_node(std::make_shared<v1::Multiply>(exp, polynomial));
        polynomial = context.mark_node(std::make_shared<v1::Add>(scalar(coefficient), product));
    }
    auto tail = context.mark_node(std::make_shared<v1::Multiply>(exp, polynomial));
    auto softplus = context.mark_node(std::make_shared<v4::SoftPlus>(scaled));
    auto negative = context.mark_node(std::make_shared<v1::Less>(scaled, boundary));
    auto stable = context.mark_node(std::make_shared<v1::Select>(negative, tail, softplus));
    auto divided = context.mark_node(std::make_shared<v1::Divide>(stable, beta));
    auto linear = context.mark_node(std::make_shared<v1::Greater>(scaled, threshold));
    Output<Node> result = context.mark_node(std::make_shared<v1::Select>(linear, input, divided));
    if (original_type.is_static() && result.get_element_type() != original_type) {
        result = context.mark_node(std::make_shared<v0::Convert>(result, original_type));
    }
    return {result};
}
}  // namespace ov::frontend::pytorch::op
