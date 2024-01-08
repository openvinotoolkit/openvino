// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "openvino/op/multiply.hpp"
//#include "openvino/frontend/type/element_type.hpp"
//#include "openvino/frontend/type/type_info.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert_like.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_mul_common(const NodeContext& context, bool inplace) {
    num_inputs_check(context, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);

    // Handle boolean inputs (convert to int64)
    if (auto dtype = context.get_input_type(0); dtype.is<type::boolean>()) {
        lhs = context.mark_node(std::make_shared<ov::op::v0::Convert>(lhs, element::i64));
    }
    if (auto dtype = context.get_input_type(1); dtype.is<type::boolean>()) {
        rhs = context.mark_node(std::make_shared<ov::op::v0::Convert>(rhs, element::i64));
    }

    // Handle inplace operation and type alignment
    if (inplace) {
        if (lhs.get_element_type().is_dynamic() || lhs.get_element_type() != rhs.get_element_type()) {
            rhs = context.mark_node(std::make_shared<v1::ConvertLike>(rhs, lhs));
        }
    } else {
        align_eltwise_input_types(context, lhs, rhs, true);
    }

    auto mul = context.mark_node(std::make_shared<v1::Multiply>(lhs, rhs));
    if (inplace) {
        context.mutate_input(0, mul);
    }
    return {mul};
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