// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "pt_framework_node.hpp"
#include "translate_session.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_set_attr(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    // It is not important which operation represent FW node
    auto fw_node = std::dynamic_pointer_cast<PtFrameworkNode>(context.get_input(0).get_node_shared_ptr());
    FRONT_END_OP_CONVERSION_CHECK(fw_node, "Input to prim::SetAttr must be FrameworkNode.");
    auto attrs = fw_node->get_attrs();
    auto name = context.get_attribute<std::string>("name");
    attrs[name] = context.get_input(1).get_any_name();
    fw_node->set_attrs(attrs);
    return {};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov