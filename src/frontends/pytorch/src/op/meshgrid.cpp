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
    std::string indexing = "ij";
    if (!context.input_is_none(1)) {
        indexing = context.const_input<std::string>(1);
    }
    auto node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
    auto attrs = node->get_attrs();
    attrs["indexing"] = indexing;
    node->set_attrs(attrs);
    return {context.mark_node(node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
