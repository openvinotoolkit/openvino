// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reshape(NodeContext& context) {
    auto shape_node = context.get_input(1).get_node();
    auto shape_node_fw_node = dynamic_cast<PtFrameworkNode*>(shape_node);
    std::shared_ptr<ov::Node> reshape;
    // TODO: move this to transform stage
    if (shape_node_fw_node && shape_node_fw_node->get_decoder()->get_op_type() == "prim::ListConstruct") {
        OutputVector inputs;
        auto axis_0 = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
        for (auto& input : shape_node->inputs()) {
            auto rank = input.get_partial_shape().rank();
            FRONT_END_OP_CONVERSION_CHECK(rank.is_dynamic() || rank.get_length() == 0, "Rank must be 0");
            auto unsqueeze = context.mark_node(std::make_shared<opset10::Unsqueeze>(input.get_source_output(), axis_0));
            inputs.push_back(unsqueeze);
        }
        auto concat = context.mark_node(std::make_shared<opset10::Concat>(inputs, 0));
        reshape = context.mark_node(std::make_shared<opset10::Reshape>(context.get_input(0), concat, false));
    } else {
        reshape =
            context.mark_node(std::make_shared<opset10::Reshape>(context.get_input(0), context.get_input(1), false));
    }
    return {reshape};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
