// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_uninitialized(const NodeContext& context) {
    auto decoder = context.get_decoder();
    auto fw_node = std::make_shared<PtFrameworkNode>(decoder, OutputVector{}, context.get_output_size());

    auto attrs = fw_node->get_attrs();
    attrs["none_value"] = "";
    attrs[PtFrameworkNode::failed_conversion_key] =
        "prim::Uninitialized represents a value that is not yet assigned. "
        "It should be removed by following transformations (e.g., If/Loop) or result in an error if actually consumed.";

    fw_node->set_attrs(attrs);
    return {context.mark_node(fw_node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov