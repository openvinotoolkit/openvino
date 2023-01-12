// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_where(NodeContext& context) {
    auto cond = context.get_input(0);
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(1), "aten::where(cond) unsupported");
    auto bool_cond = context.mark_node(std::make_shared<opset10::Convert>(cond, element::boolean));
    auto x = context.get_input(1);
    auto y = context.get_input(2);
    return {context.mark_node(std::make_shared<opset10::Select>(bool_cond, x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov