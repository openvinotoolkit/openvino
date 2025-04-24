// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs argmax(const NodeContext& node) {
    auto data = node.get_input("X");
    bool flatten = node.get_attribute<bool>("flatten");
    auto dtype = node.get_attribute<ov::element::Type>("dtype");
    const Output<ov::Node> k = ov::opset6::Constant::create(dtype, {}, {1});

    if (!flatten) {
        auto axis = node.get_attribute<int64_t>("axis");
        const auto axis_to_remove = ov::opset6::Constant::create(element::u64, Shape{}, {axis});
        auto node_topk = std::make_shared<ov::opset6::TopK>(data, k, axis, "max", "index", dtype);
        const auto reshaped_indices = std::make_shared<ov::opset6::Squeeze>(node_topk->output(1), axis_to_remove);
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Convert>(reshaped_indices, dtype)},
                                                  {"Out"});
    } else {
        int64_t axis = 0;
        const Output<ov::Node> reshape_flatten = ov::opset6::Constant::create(ov::element::i64, {1}, {-1});
        auto node_reshape = std::make_shared<ov::opset6::Reshape>(data, reshape_flatten, true);
        auto node_topk = std::make_shared<ov::opset6::TopK>(node_reshape, k, axis, "max", "index", dtype);
        const auto output_info = node.get_output_port_infos("Out");
        size_t output_size = output_info[0].second.size();
        if (output_size == 0) {
            auto out = std::make_shared<ov::opset6::Squeeze>(node_topk->output(1));
            return node.default_single_output_mapping({std::make_shared<ov::opset6::Convert>(out, dtype)}, {"Out"});
        } else {
            return node.default_single_output_mapping(
                {std::make_shared<ov::opset6::Convert>(node_topk->output(1), dtype)},
                {"Out"});
        }
    }
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
