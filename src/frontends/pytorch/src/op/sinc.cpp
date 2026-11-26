// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/sin.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_sinc(const NodeContext& context) {
    // aten::sinc(Tensor self) -> Tensor
    // aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 2);

    auto orig_x = context.get_input(0);
    auto x = get_input_with_floating_type(context, 0);

    // Promote float16 and bfloat16 inputs to float32 for higher precision and to avoid overflow
    const auto orig_type = orig_x.get_element_type();
    const bool is_half_precision = (orig_type == element::f16 || orig_type == element::bf16);
    if (is_half_precision) {
        x = context.mark_node(std::make_shared<v0::Convert>(orig_x, element::f32));
    }

    const double pi_val = 3.14159265358979323846;
    auto pi_const = context.mark_node(v0::Constant::create(element::f64, Shape{}, {pi_val}));
    pi_const = context.mark_node(std::make_shared<v1::ConvertLike>(pi_const, x));

    auto zero_const = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero_const = context.mark_node(std::make_shared<v1::ConvertLike>(zero_const, x));

    auto one_const = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));
    one_const = context.mark_node(std::make_shared<v1::ConvertLike>(one_const, x));

    // Compute pi * x
    auto pi_x = context.mark_node(std::make_shared<v1::Multiply>(x, pi_const));

    // Compute sin(pi * x)
    auto sin_pi_x = context.mark_node(std::make_shared<v0::Sin>(pi_x));

    // If x == 0, safe denominator is 1.0 to avoid 0/0 (NaN)
    auto is_zero = context.mark_node(std::make_shared<v1::Equal>(x, zero_const));
    auto safe_pi_x = context.mark_node(std::make_shared<v1::Select>(is_zero, one_const, pi_x));

    auto div = context.mark_node(std::make_shared<v1::Divide>(sin_pi_x, safe_pi_x));
    auto res = context.mark_node(std::make_shared<v1::Select>(is_zero, one_const, div));

    if (is_half_precision) {
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, orig_x));
    }

    if (!context.input_is_none(1)) {
        auto out = context.get_input(1);
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, out));
        context.mutate_input(1, res);
    }
    return {res};
}

}  // namespace ov::frontend::pytorch::op
