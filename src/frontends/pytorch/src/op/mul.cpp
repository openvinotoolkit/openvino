// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_mul_common(const NodeContext& context, bool inplace) {
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    auto lhs_dtype = simplified_type_interpret(context.get_input_type(0));
    auto rhs_dtype = simplified_type_interpret(context.get_input_type(1));

    auto left_is_bool = lhs.get_element_type() == ov::element::boolean ||
                        (lhs_dtype.is<element::Type>() && lhs_dtype.as<element::Type>() == element::boolean);
    auto right_is_bool = rhs.get_element_type() == ov::element::boolean ||
                         (rhs_dtype.is<element::Type>() && rhs_dtype.as<element::Type>() == element::boolean);

    if (left_is_bool && right_is_bool) {
        // if input types are boolean then aten::mul meand logical Add operation
        if (inplace)
            return op::inplace_translate_1to1_match_2_inputs_align_types<v1::LogicalAnd>(context);
        else
            return op::translate_1to1_match_2_inputs_align_types<v1::LogicalAnd>(context);
    }

    if (inplace)
        return op::inplace_translate_1to1_match_2_inputs_align_types<v1::Multiply>(context);
    else
        return op::translate_1to1_match_2_inputs_align_types<v1::Multiply>(context);
}

OutputVector translate_mul(const NodeContext& context) {
    return translate_mul_common(context, false);
}

OutputVector translate_mul_(const NodeContext& context) {
    return translate_mul_common(context, true);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov