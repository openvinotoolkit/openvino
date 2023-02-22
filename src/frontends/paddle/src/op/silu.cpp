// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "default_opset.hpp"
#include "openvino/frontend/paddle/node_context.hpp"
#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs rilu(const NodeContext& node) {
    auto data = node.get_input("X");
    auto one = default_opset::Constant::create(data.get_element_type(), data.get_shape(), {1});
    auto neg = std::make_shared<ov::opset6::Negative>(data);
    auto exp = std::make_shared<ov::opset6::Exp>(neg);
    auto add = std::make_shared<ov::opset6::Add>(one, exp);
    auto div = std::make_shared<ov::opset6::Divide>(data, add);
    return node.default_single_output_mapping({div}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
