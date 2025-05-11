// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs gather_nd(const NodeContext& node) {
    const auto data_node = node.get_input("X");
    const auto index_node = node.get_input("Index");
    auto shape = index_node.get_partial_shape();
    if (shape.is_static() && shape.rank().get_length() == 0)
        PADDLE_OP_CHECK(node, false, "zero 'indices' input rank is not allowed for gather_nd");
    return node.default_single_output_mapping({std::make_shared<default_opset::GatherND>(data_node, index_node)},
                                              {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
