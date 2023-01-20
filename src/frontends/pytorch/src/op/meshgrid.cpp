// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_meshgrid(NodeContext& context) {
    auto node = context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs()));
    node->set_argument(0, context.get_input_from_visible_context(0));
    if (!context.input_is_none(1)) {
        node->set_argument(1, context.get_input_from_visible_context(1));
    }
    return {node};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
