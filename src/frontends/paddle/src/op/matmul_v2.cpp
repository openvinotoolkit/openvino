// Copyright (C) 2018-2025 Intel Corporation
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

    std::shared_ptr<Node> result = mm;
    const auto output_info = node.get_output_port_infos("Out");
    size_t output_size = output_info[0].second.size();
    if (is_scalar(mm->get_output_partial_shape(0)) && output_size) {
        auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {0});
        result = std::make_shared<default_opset::Unsqueeze>(mm, unsqueeze_scalar);
    }
    return node.default_single_output_mapping({result}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
