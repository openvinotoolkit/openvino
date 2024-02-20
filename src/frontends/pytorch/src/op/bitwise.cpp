// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_not.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_xor.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_bitwise_not(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    auto not_x = context.mark_node(std::make_shared<ov::op::v13::BitwiseNot>(x));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, not_x);
    }
    return {not_x};
};

OutputVector translate_bitwise_and(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, false);
    auto and_x = context.mark_node(std::make_shared<ov::op::v13::BitwiseAnd>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, and_x);
    }
    return {and_x};
};

OutputVector translate_bitwise_or(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, false);
    auto or_x = context.mark_node(std::make_shared<ov::op::v13::BitwiseOr>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, or_x);
    }
    return {or_x};
};

OutputVector translate_bitwise_xor(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    align_eltwise_input_types(context, x, y, false);
    auto xor_x = context.mark_node(std::make_shared<ov::op::v13::BitwiseXor>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, xor_x);
    }
    return {xor_x};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
