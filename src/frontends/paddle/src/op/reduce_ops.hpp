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
    if (reduce_all) {
        std::iota(dims.begin(), dims.end(), 0);
    } else {
        dims = node.get_attribute<std::vector<int32_t>>("dim");
    }
    auto axesNode = default_opset::Constant::create(ov::element::i32, {dims.size()}, dims);
    bool scalar_output = !keep_dim;
    if (scalar_output) {
        for (int32_t i = 0; i < input_rank; i++) {
            if (std::find(dims.begin(), dims.end(), i) == dims.end()) {
                scalar_output = false;
                break;
            }
        }
    }

    auto reduceNode = std::make_shared<T>(x, axesNode, keep_dim);
    std::shared_ptr<Node> result = reduceNode;
    if (scalar_output) {
        auto unsqueeze_scalar = default_opset::Constant::create(ov::element::i64, {}, {0});
        result = std::make_shared<default_opset::Unsqueeze>(reduceNode, unsqueeze_scalar);
    }
    return node.default_single_output_mapping({result}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
