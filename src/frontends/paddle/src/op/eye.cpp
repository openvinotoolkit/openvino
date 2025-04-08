// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs eye(const NodeContext& node) {
    auto row = node.get_attribute<int64_t>("num_rows");
    auto col = node.get_attribute<int64_t>("num_columns", row);
    auto dtype = node.get_attribute<ov::element::Type>("dtype", ov::element::f32);

    const auto& row_node = std::make_shared<default_opset::Constant>(ov::element::i64, Shape{}, (row));
    const auto& col_node = std::make_shared<default_opset::Constant>(ov::element::i64, Shape{}, (col));
    const auto& diagonal_index_node = std::make_shared<default_opset::Constant>(ov::element::i32, Shape{}, (0));

    std::shared_ptr<Node> out_node;
    if (dtype == ov::element::i32 || dtype == ov::element::i64) {
        out_node = std::make_shared<default_opset::Eye>(row_node, col_node, diagonal_index_node, dtype);
    } else {
        const auto& eye_node =
            std::make_shared<default_opset::Eye>(row_node, col_node, diagonal_index_node, ov::element::i32);
        out_node = std::make_shared<default_opset::Convert>(eye_node, dtype);
    }

    return node.default_single_output_mapping({out_node}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
