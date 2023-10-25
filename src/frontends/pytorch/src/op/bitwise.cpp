// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_bitwise_not(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    FRONT_END_OP_CONVERSION_CHECK(x.get_element_type().compatible(element::boolean),
                                  "aten::bitwise_not supported only for boolean input");
    auto not_x = context.mark_node(std::make_shared<ov::op::v1::LogicalNot>(x));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, not_x);
    }
    return {not_x};
};

OutputVector translate_bitwise_and(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    FRONT_END_OP_CONVERSION_CHECK(x.get_element_type().compatible(element::boolean),
                                  "aten::bitwise_not supported only for boolean input");
    auto and_x = context.mark_node(std::make_shared<ov::op::v1::LogicalAnd>(x, y));
    return {and_x};
};

OutputVector translate_bitwise_or(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    FRONT_END_OP_CONVERSION_CHECK(x.get_element_type().compatible(element::boolean),
                                  "aten::bitwise_not supported only for boolean input");
    auto or_x = context.mark_node(std::make_shared<ov::op::v1::LogicalOr>(x, y));
    return {or_x};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
