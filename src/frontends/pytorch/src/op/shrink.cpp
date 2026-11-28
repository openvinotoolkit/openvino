// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/is_nan.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_hardshrink(const NodeContext& context) {
    // aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
    // aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    Output<Node> lambd;
    if (context.input_is_none(1)) {
        lambd = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5f}));
    } else {
        lambd = context.get_input(1);
    }
    lambd = context.mark_node(std::make_shared<v1::ConvertLike>(lambd, x));

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));

    auto abs_x = context.mark_node(std::make_shared<v0::Abs>(x));
    auto gt = context.mark_node(std::make_shared<v1::Greater>(abs_x, lambd));
    auto is_nan = context.mark_node(std::make_shared<v10::IsNaN>(x));
    auto mask = context.mark_node(std::make_shared<v1::LogicalOr>(gt, is_nan));
    auto res = context.mark_node(std::make_shared<v1::Select>(mask, x, zero));

    return {res};
}

OutputVector translate_softshrink(const NodeContext& context) {
    // aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
    // aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    Output<Node> lambd;
    if (context.input_is_none(1)) {
        lambd = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.5f}));
    } else {
        lambd = context.get_input(1);
        if (auto const_lambd = ov::as_type_ptr<v0::Constant>(lambd.get_node_shared_ptr())) {
            auto val = const_lambd->cast_vector<float>();
            if (!val.empty()) {
                PYTORCH_OP_CONVERSION_CHECK(val[0] >= 0.0f, "lambda for aten::softshrink must be non-negative");
            }
        }
    }
    lambd = context.mark_node(std::make_shared<v1::ConvertLike>(lambd, x));

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, x));

    auto neg_lambd = context.mark_node(std::make_shared<v0::Negative>(lambd));
    auto sub = context.mark_node(std::make_shared<v1::Subtract>(x, lambd));
    auto add = context.mark_node(std::make_shared<v1::Add>(x, lambd));

    auto gt = context.mark_node(std::make_shared<v1::Greater>(x, lambd));
    auto lt = context.mark_node(std::make_shared<v1::Less>(x, neg_lambd));

    auto select_pos = context.mark_node(std::make_shared<v1::Select>(gt, sub, zero));
    auto select_neg = context.mark_node(std::make_shared<v1::Select>(lt, add, select_pos));

    auto is_nan = context.mark_node(std::make_shared<v10::IsNaN>(x));
    auto res = context.mark_node(std::make_shared<v1::Select>(is_nan, x, select_neg));

    return {res};
}

}  // namespace ov::frontend::pytorch::op
