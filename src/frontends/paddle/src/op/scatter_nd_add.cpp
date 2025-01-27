// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset15.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs scatter_nd_add(const NodeContext& node) {
    auto x = node.get_input("X");
    auto index = node.get_input("Index");
    auto updates = node.get_input("Updates");
    return node.default_single_output_mapping(
        {std::make_shared<ov::opset15::ScatterNDUpdate>(x,
                                                        index,
                                                        updates,
                                                        ov::opset15::ScatterNDUpdate::Reduction::SUM)},
        {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
