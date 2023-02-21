// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs index_select(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto index = node.get_ng_input("Index");
    auto axis = node.get_attribute<int64_t>("axis");
    auto out = std::make_shared<default_opset::Gather>(data, index, axis);
    return node.default_single_output_mapping({out}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov