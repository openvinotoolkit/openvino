// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/negative.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_copysign(const NodeContext& context) {
    // aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
    // aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
    // aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
    // aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 3);

    // Promote integer inputs to floating-point
    auto lhs = get_input_with_floating_type(context, 0);
    auto rhs = get_input_with_floating_type(context, 1);
    align_eltwise_input_types(context, lhs, rhs, is_python_scalar_input(context, 1));

    auto mag = context.mark_node(std::make_shared<v0::Abs>(lhs));

    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, rhs));

    auto one = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));
    one = context.mark_node(std::make_shared<v1::ConvertLike>(one, rhs));

    // Check rhs < 0 directly (handles finite negative numbers and -inf)
    auto rhs_neg = context.mark_node(std::make_shared<v1::Less>(rhs, zero));

    // Check 1.0 / rhs < 0 to correctly capture -0.0 (where 1.0 / -0.0 = -inf < 0)
    auto inv_rhs = context.mark_node(std::make_shared<v1::Divide>(one, rhs));
    auto inv_rhs_neg = context.mark_node(std::make_shared<v1::Less>(inv_rhs, zero));

    auto is_neg = context.mark_node(std::make_shared<v1::LogicalOr>(rhs_neg, inv_rhs_neg));
    auto neg_mag = context.mark_node(std::make_shared<v0::Negative>(mag));
    Output<Node> res = context.mark_node(std::make_shared<v1::Select>(is_neg, neg_mag, mag));

    // Match destination dtype if specified by context output type (e.g. for inplace_op)
    auto out_type = context.get_output_type(0);
    if (out_type.is<element::Type>()) {
        auto dtype = out_type.as<element::Type>();
        if (dtype.is_static() && dtype != res.get_element_type()) {
            res = context.mark_node(std::make_shared<v0::Convert>(res, dtype));
        }
    }

    if (!context.input_is_none(2)) {
        auto out = context.get_input(2);
        res = context.mark_node(std::make_shared<v1::ConvertLike>(res, out));
        context.mutate_input(2, res);
    }
    return {res};
}

}  // namespace ov::frontend::pytorch::op
