// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset6.hpp"
#include "paddlepaddle_frontend/node_context.hpp"

namespace ov {
namespace frontend {
namespace pdpd {
namespace op {
NamedOutputs transpose2(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto perm = node.get_attribute<std::vector<int>>("axis");
    auto input_order = ov::opset6::Constant::create(ov::element::i64, {perm.size()}, perm);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Transpose>(data, input_order)}, {"Out"});
}

}  // namespace op
}  // namespace pdpd
}  // namespace frontend
}  // namespace ov
