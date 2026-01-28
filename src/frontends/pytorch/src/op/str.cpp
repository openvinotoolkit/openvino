// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_str(const NodeContext& context) {
    // aten::str(input) -> string
    auto input_node = context.get_input_from_visible_context(0).get_node_shared_ptr();
    auto const_node = ov::as_type_ptr<ov::op::v0::Constant>(input_node);
    
    std::string str_value;
    bool found_value = false;

    if (const_node) {
        if (const_node->get_element_type() == element::i64) {
             str_value = std::to_string(const_node->cast_vector<int64_t>()[0]);
             found_value = true;
        } else if (const_node->get_element_type() == element::f32) {
             str_value = std::to_string(const_node->cast_vector<float>()[0]);
             found_value = true;
        } else if (const_node->get_element_type() == element::boolean) {
             str_value = const_node->cast_vector<bool>()[0] ? "True" : "False";
             found_value = true;
        }
    }

    if (!found_value) {
        return {context.mark_node(std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs()))};
    }

    auto fw_node = std::make_shared<PtFrameworkNode>(context.get_decoder(), OutputVector{});
    auto attrs = fw_node->get_attrs();
    attrs["string_value"] = str_value;
    fw_node->set_attrs(attrs);
    
    return {context.mark_node(fw_node)};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
