// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs unstack(const NodeContext& node) {
    auto data = node.get_input("X");
    auto dim = node.get_attribute<int32_t>("axis");
    auto axis = std::make_shared<Constant>(ov::element::i32, Shape{}, {dim});
    auto shape = data.get_shape()
    auto num_or_sections = std::make_shared<Constant>(ov::element::i32, Shape{}, data.shape[dim]);
    auto split_outputs = std::make_shared<Split>(data, axis, num_or_sections);
    NamedOutputs named_outputs;
    auto out_names = node.get_output_names();
    auto it = std::find(out_names.begin(), out_names.end(), "Out");
    for (const auto& split_output : split_outputs) {
        named_outputs[*it].push_back(std::make_shared<Squeeze>(split_output));
    }
    return named_outputs;
}
}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
