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

using namespace ov::op;

OutputVector translate_add_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2, 3);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto dtype0 = context.get_input_type(0);
    auto dtype1 = context.get_input_type(1);
    if (dtype0.is<type::List>() && dtype1.is<type::List>()) {
        // aten::add.t(t[] a, t[] b) -> t[]
        // Case when two lists gets concatenated
        FRONT_END_OP_CONVERSION_CHECK(false, "aten::add is used for concatenation of lists, not possible to convert");
    }
    if (inplace) {
        if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type())
            rhs = context.mark_node(std::make_shared<v1::ConvertLike>(rhs, lhs));
    } else {
        align_eltwise_input_types(context, lhs, rhs, true);
    }
    if (!context.input_is_none(2)) {
        auto converted_alpha = context.mark_node(std::make_shared<v1::ConvertLike>(context.get_input(2), rhs));
        rhs = context.mark_node(std::make_shared<v1::Multiply>(converted_alpha, rhs));
    }
    auto add = context.mark_node(std::make_shared<v1::Add>(lhs, rhs));
    if (inplace)
        context.mutate_input(0, add);
    return {add};
};

OutputVector translate_add(const NodeContext& context) {
    return translate_add_common(context, false);
};

OutputVector translate_add_(const NodeContext& context) {
    return translate_add_common(context, true);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov