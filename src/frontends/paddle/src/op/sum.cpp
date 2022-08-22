// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs sum(const NodeContext& node) {
    auto datas = node.get_ng_inputs("X");
    auto data_type = datas[0].get_element_type();
    auto data_shape = datas[0].get_shape();
    std::shared_ptr<Node> out_node = datas[0].get_node_shared_ptr();
    for (int i = 1; i < datas.size(); ++i) {
        PADDLE_OP_CHECK(node,
                        data_type == datas[i].get_element_type(),
                        "sum input tensor must have the same data types!");
        PADDLE_OP_CHECK(node,
                        data_shape == datas[i].get_shape(),
                        "sum input tensor must have the same shape!");
        out_node = std::make_shared<default_opset::Add>(datas[i], out_node);
    }
    return node.default_single_output_mapping({out_node}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
