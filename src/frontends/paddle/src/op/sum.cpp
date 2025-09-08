// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs sum(const NodeContext& node) {
    auto data = node.get_ng_inputs("X");
    auto sum = data[0].get_node_shared_ptr();
    for (size_t i = 1; i < data.size(); i++) {
        sum = std::make_shared<default_opset::Add>(sum, data[i]);
    }
    return node.default_single_output_mapping({sum}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
