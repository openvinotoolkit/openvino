// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_cat(NodeContext& context) {
    // This translator is only needed to get axis as constant from external scope
    num_inputs_check(context, 2, 2);
    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{context.get_input(0)}, 1);
    auto attrs = fw_node->get_attrs();
    // If this fails it means axis is dynamic and aten::cat will be converted to fw node in regular pipeline
    attrs["axis"] = std::to_string(context.const_input<int64_t>(1));
    fw_node->set_attrs(attrs);
    return {context.mark_node(std::dynamic_pointer_cast<Node>(fw_node))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov