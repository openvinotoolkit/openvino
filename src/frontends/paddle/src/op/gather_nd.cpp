// Copyright (C) 2018-2022 Intel Corporation
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
    PADDLE_OP_CHECK(node, index_node.get_partial_shape().size() > 0, "Index shape must be non-empty");
    return node.default_single_output_mapping({std::make_shared<default_opset::GatherND>(data_node, index_node)},
                                              {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov