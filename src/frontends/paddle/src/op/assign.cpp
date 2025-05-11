// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs assign(const NodeContext& node) {
    auto x = node.get_input("X");
    auto assign = std::make_shared<default_opset::Convert>(x, x.get_element_type());

    return node.default_single_output_mapping({assign}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
