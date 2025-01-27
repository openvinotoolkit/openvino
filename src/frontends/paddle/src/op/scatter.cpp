// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs scatter(const NodeContext& node) {
    auto x = node.get_input("X");
    auto ids = node.get_input("Ids");
    auto updates = node.get_input("Updates");
    bool overwrite = node.get_attribute<bool>("overwrite");
    ov::NodeVector node_vec;
    if (ids.get_shape().size() == 0) {
        ids = std::make_shared<default_opset::Unsqueeze>(ids,
                                                         default_opset::Constant::create(ov::element::i64, {1}, {0}));
    }

    node_vec.push_back(default_opset::Constant::create(ov::element::i64, {1}, {ids.get_shape()[0]}));
    node_vec.push_back(default_opset::Constant::create(ov::element::i64, {1}, {1}));
    auto shape_node = std::make_shared<default_opset::Concat>(node_vec, 0);
    auto new_ids = std::make_shared<default_opset::Reshape>(ids, shape_node, true);
    if (overwrite) {
        return node.default_single_output_mapping({std::make_shared<ov::opset15::ScatterNDUpdate>(x, new_ids, updates)},
                                                  {"Out"});
    } else {
        auto x_dtype = x.get_element_type();
        const auto value_node = default_opset::Constant::create(x_dtype, {1}, {0});
        const auto shape_node = std::make_shared<default_opset::ShapeOf>(x);
        const auto zero_node = std::make_shared<default_opset::Broadcast>(value_node, shape_node);
        return node.default_single_output_mapping(
            {std::make_shared<ov::opset15::ScatterNDUpdate>(zero_node,
                                                            new_ids,
                                                            updates,
                                                            ov::opset15::ScatterNDUpdate::Reduction::SUM)},
            {"Out"});
    }
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov