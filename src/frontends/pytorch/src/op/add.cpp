// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/add.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_add(NodeContext& context) {
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);
    if (dtype0.is<type::List>() && dtype1.is<type::List>()) {
        // aten::add.t(t[] a, t[] b) -> t[]
        // Case when two lists gets concatenated
        return {context.mark_node(std::make_shared<ov::op::v0::Concat>(OutputVector{lhs, rhs}, 0))};
    }
    align_eltwise_input_types(context, lhs, rhs);
    if (!context.input_is_none(2)) {
        auto converted_alpha = context.mark_node(std::make_shared<ov::op::v1::ConvertLike>(context.get_input(2), rhs));
        rhs = context.mark_node(std::make_shared<ov::op::v1::Multiply>(converted_alpha, rhs));
    }
    return {context.mark_node(std::make_shared<ov::op::v1::Add>(lhs, rhs))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov