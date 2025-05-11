// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/select.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_where(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto cond = context.get_input(0);
    PYTORCH_OP_CONVERSION_CHECK(!context.input_is_none(1), "aten::where(cond) unsupported");
    auto bool_cond = context.mark_node(std::make_shared<v0::Convert>(cond, element::boolean));
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 1, 2);
    return {context.mark_node(std::make_shared<v1::Select>(bool_cond, x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
