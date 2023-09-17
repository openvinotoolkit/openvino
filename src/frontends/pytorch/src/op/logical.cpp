// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_or(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    y = context.mark_node(std::make_shared<v0::Convert>(y, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto or_node = context.mark_node(std::make_shared<v1::LogicalOr>(x, y));
    return {or_node};
};

OutputVector translate_and(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto y = context.get_input(1);
    x = context.mark_node(std::make_shared<v0::Convert>(x, element::boolean));
    y = context.mark_node(std::make_shared<v0::Convert>(y, element::boolean));
    // TODO: use bitwise op here when will be supported by openvino
    auto or_node = context.mark_node(std::make_shared<v1::LogicalAnd>(x, y));
    return {or_node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
