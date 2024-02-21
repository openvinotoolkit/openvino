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

    if (lhs.get_element_type() == ov::element::boolean && rhs.get_element_type() == ov::element::boolean) {
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