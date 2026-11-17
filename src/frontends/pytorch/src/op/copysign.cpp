// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/less.hpp"
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
    Output<Node> lhs;
    Output<Node> rhs;

    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);
    // torch.copysign returns floating-point tensor for integer inputs
    const auto dtype = lhs.get_element_type();
    if (dtype.is_static() && !dtype.is_real()) {
        auto float_type = element::f32;
        const auto out_type = simplified_type_interpret(context.get_output_type(0));
        if (out_type.is<element::Type>() && out_type.as<element::Type>().is_real()) {
            float_type = out_type.as<element::Type>();
        }
        lhs = context.mark_node(std::make_shared<v0::Convert>(lhs, float_type));
        rhs = context.mark_node(std::make_shared<v0::Convert>(rhs, float_type));
    }

    auto mag = context.mark_node(std::make_shared<v0::Abs>(lhs));
    auto zero = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0.0f}));
    zero = context.mark_node(std::make_shared<v1::ConvertLike>(zero, rhs));

    auto is_neg = context.mark_node(std::make_shared<v1::Less>(rhs, zero));
    auto neg_mag = context.mark_node(std::make_shared<v0::Negative>(mag));
    auto res = context.mark_node(std::make_shared<v1::Select>(is_neg, neg_mag, mag));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, res);
    }
    return {res};
}

}  // namespace ov::frontend::pytorch::op
