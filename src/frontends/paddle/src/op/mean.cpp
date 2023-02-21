// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/frontend/paddle/visibility.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs mean(const NodeContext& node) {
    auto x = node.get_input("X");
    auto dim = node.get_attribute<std::vector<int32_t>>("dim");
    auto keep_dim = node.get_attribute<bool>("keep_dim");
    auto reduce_all = node.get_attribute<bool>("reduce_all");
    if (reduce_all) {
        dim = std::vector<int32_t>(x.get_partial_shape().rank().get_length());
        std::iota(dim.begin(), dim.end(), 0);
    }
    auto reduce_mean = std::make_shared<default_opset::ReduceMean>(
        x,
        default_opset::Constant::create(element::i64, Shape{dim.size()}, dim),
        keep_dim);
    return node.default_single_output_mapping({reduce_mean}, {"Out"});
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
