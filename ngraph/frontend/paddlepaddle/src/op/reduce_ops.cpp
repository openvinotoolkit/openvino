// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "default_opset.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {

template <typename T>
NamedOutputs reduce_ops(const NodeContext& node) {
    auto x = node.get_ng_input("X");
    auto keep_dim = node.get_attribute<bool>("keep_dim");
    auto reduce_all = node.get_attribute<bool>("reduce_all", false);
    PDPD_OP_VALIDATION_CHECK(node, x.get_partial_shape().rank().is_static(), "reduce_ops: X rank must be static!");
    int64_t input_rank = x.get_partial_shape().rank().get_length();
    std::vector<int32_t> dims;
    if (reduce_all) {
        for (int i = 0; i < input_rank; ++i) {
            dims.push_back(i);
        }
    } else {
        dims = node.get_attribute<std::vector<int32_t>>("dim");
    }
    auto axesNode = default_opset::Constant::create(ngraph::element::i32, {dims.size()}, dims);
    return node.default_single_output_mapping({std::make_shared<T>(x, axesNode, keep_dim)}, {"Out"});
}

NamedOutputs reduce_max(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceMax>(node_context);
}

NamedOutputs reduce_mean(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceMean>(node_context);
}

NamedOutputs reduce_min(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceMin>(node_context);
}

NamedOutputs reduce_prod(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceProd>(node_context);
}

NamedOutputs reduce_sum(const NodeContext& node_context) {
    return reduce_ops<default_opset::ReduceSum>(node_context);
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
