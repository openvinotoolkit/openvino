// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/logical_xor.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_or(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    y = context.mark_node(std::make_shared<v0::Convert>(y, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto or_node = context.mark_node(std::make_shared<v1::LogicalOr>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, or_node);
    }
    return {or_node};
};

OutputVector translate_and(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    y = context.mark_node(std::make_shared<v0::Convert>(y, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto and_node = context.mark_node(std::make_shared<v1::LogicalAnd>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, and_node);
    }
    return {and_node};
};

OutputVector translate_not(const NodeContext& context) {
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto not_node = context.mark_node(std::make_shared<v1::LogicalNot>(x));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, not_node);
    }
    return {not_node};
};

OutputVector translate_xor(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    y = context.mark_node(std::make_shared<v0::Convert>(y, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto xor_node = context.mark_node(std::make_shared<v1::LogicalXor>(x, y));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, xor_node);
    }
    return {xor_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
