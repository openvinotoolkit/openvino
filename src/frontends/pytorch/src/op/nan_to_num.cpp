// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

#include <limits>

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_nan_to_num(const NodeContext& context) {
    num_inputs_check(context, 1, 4);
    auto input = context.get_input(0);

    // Get nan replacement (default: 0.0)
    Output<Node> nan_value;
    if (context.input_is_none(1)) {
        nan_value = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    } else {
        nan_value = context.get_input(1);
    }
    nan_value = context.mark_node(std::make_shared<v1::ConvertLike>(nan_value, input));

    // Get posinf replacement (default: dtype-specific finite max)
    Output<Node> posinf_value;
    if (context.input_is_none(2)) {
        std::shared_ptr<Node> posinf_node;
        switch (input.get_element_type()) {
        case element::bf16:
            posinf_node = v0::Constant::create(element::bf16, {}, {std::numeric_limits<bfloat16>::max()});
            break;
        case element::f16:
            posinf_node = v0::Constant::create(element::f16, {}, {std::numeric_limits<float16>::max()});
            break;
        case element::f64:
            posinf_node = v0::Constant::create(element::f64, {}, {std::numeric_limits<double>::max()});
            break;
        case element::f32:
            posinf_node = v0::Constant::create(element::f32, {}, {std::numeric_limits<float>::max()});
            break;
        default:
            posinf_node = v0::Constant::create(element::f32, {}, {std::numeric_limits<float>::max()});
            posinf_node = std::make_shared<v1::ConvertLike>(posinf_node, input);
        }
        posinf_value = context.mark_node(posinf_node);
    } else {
        posinf_value = context.get_input(2);
        posinf_value = context.mark_node(std::make_shared<v1::ConvertLike>(posinf_value, input));
    }

    // Get neginf replacement (default: dtype-specific finite lowest)
    Output<Node> neginf_value;
    if (context.input_is_none(3)) {
        std::shared_ptr<Node> neginf_node;
        switch (input.get_element_type()) {
        case element::bf16:
            neginf_node = v0::Constant::create(element::bf16, {}, {std::numeric_limits<bfloat16>::lowest()});
            break;
        case element::f16:
            neginf_node = v0::Constant::create(element::f16, {}, {std::numeric_limits<float16>::lowest()});
            break;
        case element::f64:
            neginf_node = v0::Constant::create(element::f64, {}, {std::numeric_limits<double>::lowest()});
            break;
        case element::f32:
            neginf_node = v0::Constant::create(element::f32, {}, {std::numeric_limits<float>::lowest()});
            break;
        default:
            neginf_node = v0::Constant::create(element::f32, {}, {std::numeric_limits<float>::lowest()});
            neginf_node = std::make_shared<v1::ConvertLike>(neginf_node, input);
        }
        neginf_value = context.mark_node(neginf_node);
    } else {
        neginf_value = context.get_input(3);
        neginf_value = context.mark_node(std::make_shared<v1::ConvertLike>(neginf_value, input));
    }

    // Create masks for NaN and Inf
    auto is_nan = context.mark_node(std::make_shared<v10::IsNaN>(input));
    auto is_inf = context.mark_node(std::make_shared<v10::IsInf>(input));

    // Create zero for sign comparison
    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input));

    // Explicit sign checks
    auto is_positive = context.mark_node(std::make_shared<v1::Greater>(input, zero));
    auto is_negative = context.mark_node(std::make_shared<v1::Less>(input, zero));

    auto is_posinf = context.mark_node(std::make_shared<v1::LogicalAnd>(is_inf, is_positive));
    auto is_neginf = context.mark_node(std::make_shared<v1::LogicalAnd>(is_inf, is_negative));

    // Apply replacements via Select chain
    auto result = context.mark_node(std::make_shared<v1::Select>(is_nan, nan_value, input));
    result = context.mark_node(std::make_shared<v1::Select>(is_posinf, posinf_value, result));
    result = context.mark_node(std::make_shared<v1::Select>(is_neginf, neginf_value, result));

    return {result};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
