// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {
using namespace ov::op;

OutputVector translate_logaddexp2(const NodeContext& context) {
    // aten::logaddexp2(Tensor self, Tensor other, *, Tensor? out=None) -> Tensor
    num_inputs_check(context, 2, 3);
    Output<Node> input1;
    Output<Node> input2;
    std::tie(input1, input2) = get_inputs_with_promoted_types(context, 0, 1);
    const auto input_type = input1.get_element_type();
    if (input_type.is_static() && !input_type.is_real()) {
        input1 = context.mark_node(std::make_shared<v0::Convert>(input1, element::f32));
        input2 = context.mark_node(std::make_shared<v0::Convert>(input2, element::f32));
    }

    // log2(2^x + 2^y) = max(x, y) + log2(2^(x-max) + 2^(y-max)).
    // Scaling the differences before Exp avoids overflow for large inputs.
    auto ln2 = context.mark_node(v0::Constant::create(element::f64, Shape{}, {0.69314718055994530942}));
    ln2 = context.mark_node(std::make_shared<v1::ConvertLike>(ln2, input1));
    auto max_val = context.mark_node(std::make_shared<v1::Maximum>(input1, input2));
    auto diff1 = context.mark_node(std::make_shared<v1::Subtract>(input1, max_val));
    auto diff2 = context.mark_node(std::make_shared<v1::Subtract>(input2, max_val));
    auto scaled_diff1 = context.mark_node(std::make_shared<v1::Multiply>(diff1, ln2));
    auto scaled_diff2 = context.mark_node(std::make_shared<v1::Multiply>(diff2, ln2));
    auto exp1 = context.mark_node(std::make_shared<v0::Exp>(scaled_diff1));
    auto exp2 = context.mark_node(std::make_shared<v0::Exp>(scaled_diff2));
    auto sum = context.mark_node(std::make_shared<v1::Add>(exp1, exp2));
    auto log_sum = context.mark_node(std::make_shared<v0::Log>(sum));
    auto log2_sum = context.mark_node(std::make_shared<v1::Divide>(log_sum, ln2));
    auto finite_result = context.mark_node(std::make_shared<v1::Add>(log2_sum, max_val));

    // Avoid inf - inf for equal infinite operands, and preserve NaN propagation.
    auto max_is_inf = context.mark_node(std::make_shared<v10::IsInf>(max_val));
    auto infinite_result = context.mark_node(std::make_shared<v1::Select>(max_is_inf, max_val, finite_result));
    auto input1_is_nan = context.mark_node(std::make_shared<v10::IsNaN>(input1));
    auto input2_is_nan = context.mark_node(std::make_shared<v10::IsNaN>(input2));
    auto input_has_nan = context.mark_node(std::make_shared<v1::LogicalOr>(input1_is_nan, input2_is_nan));
    auto nan_value = context.mark_node(std::make_shared<v1::Add>(input1, input2));
    auto result = context.mark_node(std::make_shared<v1::Select>(input_has_nan, nan_value, infinite_result));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, result);
    }
    return {result};
}
}  // namespace ov::frontend::pytorch::op
