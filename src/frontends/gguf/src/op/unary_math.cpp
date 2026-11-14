// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

namespace {

// Evaluate `body` in f32, ggml's reference precision for these ops (1/6 or exp(x) - 1 for small x are not
// representable in f16), and return the operand's element type.
template <typename Body>
ov::Output<ov::Node> in_f32(const ov::Output<ov::Node>& input, Body body) {
    const auto type = input.get_element_type();
    ov::Output<ov::Node> x = input;
    if (type != ov::element::f32) {
        x = std::make_shared<ov::op::v0::Convert>(x, ov::element::f32);
    }
    ov::Output<ov::Node> res = body(x);
    if (type != ov::element::f32) {
        res = std::make_shared<ov::op::v0::Convert>(res, type);
    }
    return res;
}

}  // namespace

// GGML_UNARY_OP_HARDSIGMOID: clamp(x / 6 + 0.5, 0, 1).
OutputVector translate_unary_hardsigmoid(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = in_f32(context.get_input(0), [](const ov::Output<ov::Node>& x) -> ov::Output<ov::Node> {
        return std::make_shared<ov::op::v0::HardSigmoid>(
            x,
            ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f / 6.0f}),
            ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f}));
    });
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_UNARY_OP_EXPM1: exp(x) - 1.
OutputVector translate_unary_expm1(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res = in_f32(context.get_input(0), [](const ov::Output<ov::Node>& x) -> ov::Output<ov::Node> {
        const auto one = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f});
        auto half =
            std::make_shared<ov::op::v1::Multiply>(x, ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f}));
        auto exp_half = std::make_shared<ov::op::v0::Exp>(half);
        // Splitting positive exponents avoids premature overflow in exp kernels.
        auto exponential =
            std::make_shared<ov::op::v1::Select>(std::make_shared<ov::op::v1::Greater>(x, one),
                                                 std::make_shared<ov::op::v1::Multiply>(exp_half, exp_half),
                                                 std::make_shared<ov::op::v0::Exp>(x));
        ov::Output<ov::Node> polynomial = ov::op::v0::Constant::create(ov::element::f32, {}, {1.0f / 720.0f});
        for (const float coefficient : {1.0f / 120.0f, 1.0f / 24.0f, 1.0f / 6.0f, 0.5f}) {
            polynomial =
                std::make_shared<ov::op::v1::Add>(ov::op::v0::Constant::create(ov::element::f32, {}, {coefficient}),
                                                  std::make_shared<ov::op::v1::Multiply>(x, polynomial));
        }
        auto small_result = std::make_shared<ov::op::v1::Multiply>(
            x,
            std::make_shared<ov::op::v1::Add>(one, std::make_shared<ov::op::v1::Multiply>(x, polynomial)));
        return std::make_shared<ov::op::v1::Select>(
            std::make_shared<ov::op::v1::Less>(std::make_shared<ov::op::v0::Abs>(x),
                                               ov::op::v0::Constant::create(ov::element::f32, {}, {0.125f})),
            small_result,
            std::make_shared<ov::op::v1::Subtract>(exponential, one));
    });
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_UNARY_OP_STEP: 1 where x > 0, else 0.
OutputVector translate_unary_step(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);
    auto zero = ov::op::v0::Constant::create(input.get_element_type(), {}, {0.0f});
    auto res = std::make_shared<ov::op::v0::Convert>(std::make_shared<ov::op::v1::Greater>(input, zero),
                                                     input.get_element_type());
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_UNARY_OP_ROUND: round half away from zero, like roundf.
OutputVector translate_unary_round(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto res =
        std::make_shared<ov::op::v5::Round>(context.get_input(0), ov::op::v5::Round::RoundMode::HALF_AWAY_FROM_ZERO);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
