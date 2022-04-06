// Copyright (C) 2018-2022 Intel Corporation
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
    auto axesNode = default_opset::Constant::create(ngraph::element::i32, {dims.size()}, dims);
    return node.default_single_output_mapping({std::make_shared<T>(x, axesNode, keep_dim)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov