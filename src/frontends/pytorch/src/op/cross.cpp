// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/roll.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> translate_cross_base(const NodeContext& context, Output<Node> self, Output<Node> other, Output<Node> dim) {
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {2}));
    auto x_roll_1 = context.mark_node(std::make_shared<v7::Roll>(self, const_2, dim));
    auto x_roll_2 = context.mark_node(std::make_shared<v7::Roll>(self, const_1, dim));
    auto y_roll_1 = context.mark_node(std::make_shared<v7::Roll>(other, const_1, dim));
    auto y_roll_2 = context.mark_node(std::make_shared<v7::Roll>(other, const_2, dim));
    auto mul_1 = context.mark_node(std::make_shared<v1::Multiply>(x_roll_1, y_roll_1));
    auto mul_2 = context.mark_node(std::make_shared<v1::Multiply>(x_roll_2, y_roll_2));
    return context.mark_node(std::make_shared<v1::Subtract>(mul_1, mul_2));
}

}  // namespace
OutputVector translate_linalg_cross(const NodeContext& context) {
    // aten::linalg_cross(Tensor self, Tensor other, int? dim=-1) -> Tensor
    // aten::linalg_cross.out(Tensor self, Tensor other, int? dim=-1, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    Output<Node> self;
    Output<Node> other;
    std::tie(self, other) = get_inputs_with_promoted_types(context, 0, 1);
    auto const_minus_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    Output<Node> dim;
    if (context.input_is_none(2)) {
        dim = const_minus_1;
    } else {
        dim = context.get_input(2);
        auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        dim = context.mark_node(std::make_shared<v0::Unsqueeze>(dim, const_0));
    }
    auto res = translate_cross_base(context, self, other, dim);
    if (!context.input_is_none(3)) {
        context.mutate_input(3, res);
    }

    return {res};
};

OutputVector translate_cross(const NodeContext& context) {
    // aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
    // aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 3, 4);
    Output<Node> self;
    Output<Node> other;
    std::tie(self, other) = get_inputs_with_promoted_types(context, 0, 1);
    Output<Node> dim;
    if (context.input_is_none(2)) {
        //  If dim is not given, it defaults to the first dimension found with the size 3
        auto pshape = self.get_partial_shape();
        if (pshape.rank().is_dynamic()) {
            FRONT_END_GENERAL_CHECK(false, "Rank should be known for aten::cross without explicit dim");
        }
        size_t dim_id = static_cast<size_t>(pshape.rank().get_length());
        size_t rank = static_cast<size_t>(pshape.rank().get_length());
        for (size_t i = 0; i < rank; i++) {
            if (pshape[i].is_static() && pshape[i] == ov::Dimension(3)) {
                dim_id = i;
                break;
            }
        }
        if (dim_id == rank) {
            FRONT_END_GENERAL_CHECK(false, "Suitable dim for aten::cross not found");
        }
        dim = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {dim_id}));

    } else {
        dim = context.get_input(2);
        auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
        dim = context.mark_node(std::make_shared<v0::Unsqueeze>(dim, const_0));
    }
    auto res = translate_cross_base(context, self, other, dim);
    if (!context.input_is_none(3)) {
        context.mutate_input(3, res);
    }

    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
