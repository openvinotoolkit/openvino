// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {

template <typename T>
NamedOutputs reduce_ops(const NodeContext& node) {
    auto x = node.get_input("X");
    auto keep_dim = node.get_attribute<bool>("keep_dim");
    auto reduce_all = node.get_attribute<bool>("reduce_all", false);

    PADDLE_OP_CHECK(node, x.get_partial_shape().rank().is_static(), "reduce_ops: X rank must be static!");
    int64_t input_rank = x.get_partial_shape().rank().get_length();
    std::vector<int32_t> dims(input_rank);

    auto any = node.get_attribute_as_any("dim");
    if (any.is<std::vector<int32_t>>()) {
        dims = node.get_attribute<std::vector<int32_t>>("dim");
    } else {
        auto dim = any.as<std::vector<int64_t>>();
        dims = std::vector<int32_t>(dim.begin(), dim.end());
    }

    size_t axis_size = dims.size();
    reduce_all = reduce_all || (axis_size == input_rank || axis_size == 0);

    if (reduce_all) {
        dims = std::vector<int32_t>(input_rank);
        std::iota(dims.begin(), dims.end(), 0);
    }

    auto axes_node = default_opset::Constant::create(ov::element::i32, {dims.size()}, dims);
    bool scalar_output = !keep_dim;
    if (scalar_output) {
        for (int32_t i = 0; i < input_rank; i++) {
            if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
                scalar_output = false;
                break;
            }
        }
    }

    auto reduce_node = std::make_shared<T>(x, axes_node, keep_dim);
    std::shared_ptr<Node> result = reduce_node;
    if (scalar_output) {
        auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {0});
        result = std::make_shared<default_opset::Unsqueeze>(reduce_node, unsqueeze_scalar);
    }

    const auto output_info = node.get_output_port_infos("Out");
    size_t output_size = output_info[0].second.size();
    if (reduce_all && !output_size) {
        result = std::make_shared<default_opset::Squeeze>(reduce_node);
    }
    return node.default_single_output_mapping({result}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
