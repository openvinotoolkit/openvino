// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_device(NodeContext& context) {
    auto decoder = context.get_decoder();
    auto node = std::make_shared<PtFrameworkNode>(context.get_decoder(), context.inputs());
    auto attrs = node->get_attrs();
    auto device = decoder->get_device();
    attrs["string_value"] = device;
    node->set_attrs(attrs);
    return {context.mark_node(node)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
