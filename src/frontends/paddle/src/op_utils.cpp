// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

using namespace ov::frontend::paddle::op::default_opset;
using namespace ov;
using namespace ov::frontend;

namespace ov {
namespace frontend {
namespace paddle {
Output<Node> get_tensor_list(const OutputVector& node) {
    auto tensor_list = node;
    for (size_t i = 0; i < tensor_list.size(); i++) {
        if (tensor_list[i].get_partial_shape().rank().get_length() == 0) {
            tensor_list[i] = std::make_shared<op::default_opset::Unsqueeze>(
                tensor_list[i],
                op::default_opset::Constant::create(element::i64, {1}, {0}));
        }
    }
    Output<Node> res;
    if (node.size() == 1) {
        res = tensor_list[0];
    } else {
        res = std::make_shared<op::default_opset::Concat>(tensor_list, 0);
    }
    return res;
}

Output<Node> get_tensor_safe(const Output<Node>& node) {
    auto node_dim = node.get_partial_shape().rank().get_length();
    if (node_dim == 0) {
        return std::make_shared<op::default_opset::Unsqueeze>(
            node,
            op::default_opset::Constant::create(element::i32, {1}, {0}));
    } else {
        return node;
    }
}

}  // namespace paddle
}  // namespace frontend
}  // namespace ov