// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reverse.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_reverse(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto data = context.get_input(0);

    Output<Node> axes;
    if (context.has_attribute("dim")) {
        // Check if it's a list of dims or a single dim
        auto dims = context.get_attribute<std::vector<int64_t>>("dim");
        axes = v0::Constant::create(element::i64, Shape{dims.size()}, dims);
    } else if (!context.input_is_none(1)) {
        axes = context.get_input(1);
    } else {
        FRONT_END_OP_CONVERSION_CHECK(
            false,
            "Reverse/flip operation requires 'dim' attribute or second input to specify axes");
    }

    axes = context.mark_node(std::make_shared<v0::Convert>(axes, element::i64));
    auto reverse_node = std::make_shared<v1::Reverse>(data, axes, v1::Reverse::Mode::INDEX);
    return {context.mark_node(reverse_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov