// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov::frontend::paddle::op {
NamedOutputs argmax(const NodeContext& node) {
    auto data = node.get_input("X");
    bool flatten = node.get_attribute<bool>("flatten");
    auto dtype = node.get_attribute<ov::element::Type>("dtype");
    const Output<ov::Node> k = ov::opset6::Constant::create(dtype, {}, {1});

    if (!flatten) {
        auto axis = node.get_attribute<int64_t>("axis");
        auto node_topk = std::make_shared<ov::opset6::TopK>(data, k, axis, "max", "index", dtype);
        auto result = node_topk->output(1);
        // Paddle drops the reduced axis only when keepdims is false; with keepdims == true the
        // TopK index output already keeps the axis (size k==1) at the correct position, so the
        // Squeeze must be skipped.
        if (!node.get_attribute<bool>("keepdims", false)) {
            const auto axis_to_remove = ov::opset6::Constant::create(element::u64, Shape{}, {axis});
            result = std::make_shared<ov::opset6::Squeeze>(result, axis_to_remove);
        }
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Convert>(result, dtype)}, {"Out"});
    } else {
        int64_t axis = 0;
        const Output<ov::Node> reshape_flatten = ov::opset6::Constant::create(ov::element::i64, {1}, {-1});
        auto node_reshape = std::make_shared<ov::opset6::Reshape>(data, reshape_flatten, true);
        auto node_topk = std::make_shared<ov::opset6::TopK>(node_reshape, k, axis, "max", "index", dtype);
        // TopK keeps the reduced axis as a size-1 dimension, so the index output is a 1-D tensor of
        // one element; the shape Paddle expects is restored from the output port information.
        Output<ov::Node> result = node_topk->output(1);
        const auto output_info = node.get_output_port_infos("Out");
        size_t output_size = output_info[0].second.size();
        if (output_size == 0) {
            // Paddle returns a scalar when keepdims is false.
            result = std::make_shared<ov::opset6::Squeeze>(result);
        } else if (output_size > 1) {
            // With keepdims == true the flattened reduction keeps the rank of the input, every
            // dimension being of size 1, so an (A, B, C) input yields a (1, 1, 1) output.
            const std::vector<int64_t> keepdims_shape(output_size, 1);
            result = std::make_shared<ov::opset6::Reshape>(
                result,
                ov::opset6::Constant::create(ov::element::i64, Shape{output_size}, keepdims_shape),
                false);
        }
        return node.default_single_output_mapping({std::make_shared<ov::opset6::Convert>(result, dtype)}, {"Out"});
    }
}

}  // namespace ov::frontend::paddle::op
