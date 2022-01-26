// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs matmul_v2(const NodeContext& node) {
    const auto x = node.get_input("X");
    const auto y = node.get_input("Y");
    const auto transpose_a = node.get_attribute<bool>("trans_x", false);
    const auto transpose_b = node.get_attribute<bool>("trans_y", false);
    const auto mm = std::make_shared<default_opset::MatMul>(x, y, transpose_a, transpose_b);
    return node.default_single_output_mapping({mm}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
