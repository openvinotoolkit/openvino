// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/is_inf.hpp"
#include "openvino/op/log.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_expm1(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    // aten::expm1(Tensor self) -> Tensor
    // aten::expm1(Tensor self, Tensor out) -> out Tensor

    auto exp_vector = translate_1to1_match_1_inputs_with_fp32_type_alignment<v0::Exp>(context);
    PYTORCH_OP_CONVERSION_CHECK(exp_vector.size() == 1,
                                "Expected exactly one element in the vector. Got: ",
                                exp_vector.size());
    auto exp = exp_vector[0];
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    const_1 = context.mark_node(std::make_shared<v1::ConvertLike>(const_1, exp));
    auto exp_minus_one = context.mark_node(std::make_shared<v1::Subtract>(exp, const_1));
    // exp(x) - 1 loses all precision for small x, because exp(x) rounds to 1. Compensate the rounding error of
    // u = exp(x) as (u - 1) * (x / log(u)), and return x itself where u rounds to exactly 1.
    const auto x = exp.get_node_shared_ptr()->input_value(0);
    auto log = context.mark_node(std::make_shared<v0::Log>(exp));
    auto scale = context.mark_node(std::make_shared<v1::Divide>(x, log));
    Output<Node> expm1 = context.mark_node(std::make_shared<v1::Multiply>(exp_minus_one, scale));
    auto u_is_one = context.mark_node(std::make_shared<v1::Equal>(exp, const_1));
    expm1 = context.mark_node(std::make_shared<v1::Select>(u_is_one, x, expm1));
    // For large negative x, exp(x) - 1 rounds to exactly -1, which is already the correct result. There u may be
    // subnormal or 0, where log(u) is imprecise or -inf and the formula above is wrong. Where exp(x) overflows to
    // +inf, the formula gives nan while exp(x) - 1 is +inf.
    auto const_minus_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    const_minus_1 = context.mark_node(std::make_shared<v1::ConvertLike>(const_minus_1, exp));
    auto is_minus_one = context.mark_node(std::make_shared<v1::Equal>(exp_minus_one, const_minus_1));
    auto exp_is_inf = context.mark_node(std::make_shared<v10::IsInf>(exp));
    auto keep_exp_minus_one = context.mark_node(std::make_shared<v1::LogicalOr>(is_minus_one, exp_is_inf));
    expm1 = context.mark_node(std::make_shared<v1::Select>(keep_exp_minus_one, exp_minus_one, expm1));

    if (!context.input_is_none(1)) {
        context.mutate_input(1, expm1);
    }
    return {expm1};
}

}  // namespace ov::frontend::pytorch::op